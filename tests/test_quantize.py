"""Test harness for bounding box quantization.

Tests quantize_boxes() and its helpers against synthetic data that mimics
real YOLO detection output on UI screenshots. Compares metrics before and
after quantization to verify improvement.

No ML model dependencies required -- operates purely on box coordinates.

The quantization functions are imported via a lightweight shim to avoid
pulling in the heavy ML dependencies from util/utils.py.
"""
import copy
import math
import random
import sys
import os
import types
import ast
import textwrap
import pytest


# ---------------------------------------------------------------------------
# Extract only the quantization functions from utils.py without importing
# the full module (which requires CUDA, ultralytics, easyocr, etc.)
# ---------------------------------------------------------------------------

def _load_quantize_functions():
    """Parse util/utils.py and exec only the quantization functions."""
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'util', 'utils.py')
    with open(utils_path) as f:
        source = f.read()

    tree = ast.parse(source)
    target_funcs = {'_cluster_1d', '_cluster_sizes', 'quantize_boxes'}
    func_sources = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_funcs:
            func_sources.append(ast.get_source_segment(source, node))

    module = types.ModuleType('quantize_funcs')
    module.__dict__['copy'] = copy
    for src in func_sources:
        exec(compile(src, '<quantize>', 'exec'), module.__dict__)

    return module


_qmod = _load_quantize_functions()
_cluster_1d = _qmod._cluster_1d
_cluster_sizes = _qmod._cluster_sizes
quantize_boxes = _qmod.quantize_boxes


# ---------------------------------------------------------------------------
# Helpers for building synthetic data and computing metrics
# ---------------------------------------------------------------------------

def make_icon_box(x1, y1, x2, y2):
    """Create an icon-type box element (normalized coords)."""
    return {'type': 'icon', 'bbox': [x1, y1, x2, y2],
            'interactivity': True, 'content': None, 'source': 'box_yolo_content_yolo'}


def make_text_box(x1, y1, x2, y2, content='hello'):
    """Create a text-type box element (normalized coords)."""
    return {'type': 'text', 'bbox': [x1, y1, x2, y2],
            'interactivity': False, 'content': content, 'source': 'box_ocr_content_ocr'}


def add_jitter(val, max_jitter_px, dim):
    """Add random pixel-level jitter to a normalized coordinate."""
    jitter = random.uniform(-max_jitter_px, max_jitter_px) / dim
    return max(0.0, min(1.0, val + jitter))


def make_toolbar_icons(n, x_start, y_top, icon_size_px, spacing_px, w, h, jitter_px=3):
    """Generate a row of n icons with realistic YOLO jitter.

    Returns (ground_truth_boxes, jittered_boxes) as lists of icon dicts.
    Ground truth has exact pixel-aligned coords; jittered simulates YOLO output.
    """
    gt_boxes = []
    jittered_boxes = []
    random.seed(42)
    for i in range(n):
        # Ground truth: perfectly aligned
        x1 = (x_start + i * (icon_size_px + spacing_px)) / w
        y1 = y_top / h
        x2 = (x_start + i * (icon_size_px + spacing_px) + icon_size_px) / w
        y2 = (y_top + icon_size_px) / h
        gt_boxes.append(make_icon_box(x1, y1, x2, y2))

        # Jittered: add small random perturbation
        jx1 = add_jitter(x1, jitter_px, w)
        jy1 = add_jitter(y1, jitter_px, h)
        jx2 = add_jitter(x2, jitter_px, w)
        jy2 = add_jitter(y2, jitter_px, h)
        jittered_boxes.append(make_icon_box(jx1, jy1, jx2, jy2))

    return gt_boxes, jittered_boxes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def edge_variance(boxes, axis, dim):
    """Variance of a specific edge coordinate (in pixels) across icon boxes."""
    vals = [b['bbox'][axis] * dim for b in boxes if b['type'] == 'icon']
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def size_variance(boxes, w, h):
    """Variance of (width, height) in pixels across icon boxes."""
    sizes = []
    for b in boxes:
        if b['type'] != 'icon':
            continue
        bw = (b['bbox'][2] - b['bbox'][0]) * w
        bh = (b['bbox'][3] - b['bbox'][1]) * h
        sizes.append((bw, bh))
    if len(sizes) < 2:
        return 0.0, 0.0
    mean_w = sum(s[0] for s in sizes) / len(sizes)
    mean_h = sum(s[1] for s in sizes) / len(sizes)
    var_w = sum((s[0] - mean_w) ** 2 for s in sizes) / len(sizes)
    var_h = sum((s[1] - mean_h) ** 2 for s in sizes) / len(sizes)
    return var_w, var_h


def max_aspect_deviation(boxes, w, h):
    """Max deviation from 1:1 aspect ratio across near-square icon boxes."""
    max_dev = 0.0
    for b in boxes:
        if b['type'] != 'icon':
            continue
        bw = (b['bbox'][2] - b['bbox'][0]) * w
        bh = (b['bbox'][3] - b['bbox'][1]) * h
        if min(bw, bh) > 0:
            ratio = max(bw, bh) / min(bw, bh)
            if ratio < 1.15:  # only near-square boxes
                max_dev = max(max_dev, abs(ratio - 1.0))
    return max_dev


def mean_distance_to_gt(boxes, gt_boxes, w, h):
    """Mean L2 distance in pixels between corresponding boxes and ground truth."""
    assert len(boxes) == len(gt_boxes)
    total = 0.0
    for b, gt in zip(boxes, gt_boxes):
        dist = 0.0
        for axis, dim in [(0, w), (1, h), (2, w), (3, h)]:
            diff = (b['bbox'][axis] - gt['bbox'][axis]) * dim
            dist += diff ** 2
        total += math.sqrt(dist)
    return total / len(boxes)


# ---------------------------------------------------------------------------
# Tests for _cluster_1d
# ---------------------------------------------------------------------------

class TestCluster1D:
    def test_empty(self):
        assert _cluster_1d([], 5) == []

    def test_single_value(self):
        result = _cluster_1d([(10.0, 0)], 5)
        assert len(result) == 1
        assert len(result[0]) == 1

    def test_all_within_tolerance(self):
        vals = [(10.0, 0), (12.0, 1), (14.0, 2), (11.0, 3)]
        result = _cluster_1d(vals, 5)
        assert len(result) == 1
        assert len(result[0]) == 4

    def test_two_clusters(self):
        vals = [(10.0, 0), (12.0, 1), (50.0, 2), (52.0, 3)]
        result = _cluster_1d(vals, 5)
        assert len(result) == 2
        indices_0 = {idx for _, idx in result[0]}
        indices_1 = {idx for _, idx in result[1]}
        assert indices_0 == {0, 1}
        assert indices_1 == {2, 3}

    def test_exact_tolerance_boundary(self):
        vals = [(10.0, 0), (15.0, 1), (20.0, 2)]
        result = _cluster_1d(vals, 5)
        # 10->15 within 5, 15->20 within 5, so all merge
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests for _cluster_sizes
# ---------------------------------------------------------------------------

class TestClusterSizes:
    def test_empty(self):
        assert _cluster_sizes([], 5) == []

    def test_single(self):
        result = _cluster_sizes([(32, 32, 0)], 5)
        assert len(result) == 1

    def test_similar_sizes_cluster(self):
        sizes = [(30, 31, 0), (32, 33, 1), (31, 30, 2), (33, 32, 3)]
        result = _cluster_sizes(sizes, 5)
        assert len(result) == 1
        assert len(result[0]) == 4

    def test_different_sizes_separate(self):
        sizes = [(30, 30, 0), (31, 31, 1), (60, 60, 2), (61, 61, 3)]
        result = _cluster_sizes(sizes, 5)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests for quantize_boxes
# ---------------------------------------------------------------------------

class TestQuantizeBoxes:
    W, H = 1920, 1080  # typical desktop resolution

    def test_empty_boxes(self):
        result = quantize_boxes([], self.W, self.H)
        assert result == []

    def test_text_boxes_unchanged(self):
        text = make_text_box(0.1, 0.2, 0.3, 0.4, 'hello')
        original = copy.deepcopy(text)
        result = quantize_boxes([text], self.W, self.H)
        assert result[0]['bbox'] == original['bbox']
        assert result[0]['type'] == 'text'

    def test_single_icon_pixel_snapped(self):
        """Single icon should at least get pixel-snapped."""
        box = make_icon_box(0.10052, 0.20037, 0.15078, 0.25019)
        result = quantize_boxes([box], self.W, self.H)
        for axis, dim in [(0, self.W), (1, self.H), (2, self.W), (3, self.H)]:
            pixel_val = result[0]['bbox'][axis] * dim
            assert abs(pixel_val - round(pixel_val)) < 1e-9, \
                f"axis {axis}: {pixel_val} not pixel-snapped"

    def test_does_not_mutate_input(self):
        boxes = [make_icon_box(0.1, 0.2, 0.15, 0.25)]
        original_bbox = boxes[0]['bbox'][:]
        quantize_boxes(boxes, self.W, self.H)
        assert boxes[0]['bbox'] == original_bbox

    def test_edge_alignment_toolbar(self):
        """Icons in a toolbar row should have identical y1 and y2 after snapping."""
        _, jittered = make_toolbar_icons(6, x_start=100, y_top=50,
                                         icon_size_px=32, spacing_px=8,
                                         w=self.W, h=self.H, jitter_px=3)
        result = quantize_boxes(jittered, self.W, self.H, snap_tolerance_px=5)

        y1_vals = [b['bbox'][1] for b in result]
        assert len(set(y1_vals)) == 1, f"y1 values not aligned: {y1_vals}"

        y2_vals = [b['bbox'][3] for b in result]
        assert len(set(y2_vals)) == 1, f"y2 values not aligned: {y2_vals}"

    def test_size_uniformity_toolbar(self):
        """Icons of the same logical size should become identical after quantization."""
        _, jittered = make_toolbar_icons(6, x_start=100, y_top=50,
                                         icon_size_px=32, spacing_px=8,
                                         w=self.W, h=self.H, jitter_px=3)
        result = quantize_boxes(jittered, self.W, self.H, snap_tolerance_px=5)

        widths = set()
        heights = set()
        for b in result:
            bw = round((b['bbox'][2] - b['bbox'][0]) * self.W, 6)
            bh = round((b['bbox'][3] - b['bbox'][1]) * self.H, 6)
            widths.add(bw)
            heights.add(bh)

        assert len(widths) == 1, f"widths not uniform: {widths}"
        assert len(heights) == 1, f"heights not uniform: {heights}"

    def test_aspect_ratio_normalized(self):
        """Near-square icons should become exactly square."""
        boxes = [
            make_icon_box(0.1, 0.1, 0.1 + 31/1920, 0.1 + 33/1080),
            make_icon_box(0.3, 0.1, 0.3 + 33/1920, 0.1 + 31/1080),
        ]
        result = quantize_boxes(boxes, self.W, self.H, snap_tolerance_px=5)

        for b in result:
            bw = (b['bbox'][2] - b['bbox'][0]) * self.W
            bh = (b['bbox'][3] - b['bbox'][1]) * self.H
            assert abs(bw - bh) < 0.01, f"Not square: {bw} x {bh}"

    def test_mixed_text_and_icons(self):
        """Text boxes should pass through; icons should be quantized."""
        text = make_text_box(0.5, 0.5, 0.7, 0.55, 'Save')
        _, icons = make_toolbar_icons(4, x_start=100, y_top=50,
                                       icon_size_px=32, spacing_px=8,
                                       w=self.W, h=self.H, jitter_px=3)
        boxes = [text] + icons
        original_text_bbox = text['bbox'][:]

        result = quantize_boxes(boxes, self.W, self.H, snap_tolerance_px=5)

        assert result[0]['bbox'] == original_text_bbox
        assert result[0]['type'] == 'text'

        icon_results = [b for b in result if b['type'] == 'icon']
        y1_vals = [b['bbox'][1] for b in icon_results]
        assert len(set(y1_vals)) == 1

    def test_boxes_stay_within_bounds(self):
        """All coordinates should remain in [0, 1] after quantization."""
        boxes = [
            make_icon_box(0.001, 0.001, 0.02, 0.02),
            make_icon_box(0.98, 0.98, 0.999, 0.999),
        ]
        result = quantize_boxes(boxes, self.W, self.H, snap_tolerance_px=5)
        for b in result:
            for v in b['bbox']:
                assert 0.0 <= v <= 1.0, f"Out of bounds: {v}"

    def test_valid_box_dimensions(self):
        """x2 > x1 and y2 > y1 must hold for all output boxes."""
        _, jittered = make_toolbar_icons(8, x_start=50, y_top=30,
                                         icon_size_px=24, spacing_px=4,
                                         w=self.W, h=self.H, jitter_px=4)
        result = quantize_boxes(jittered, self.W, self.H, snap_tolerance_px=8)
        for b in result:
            assert b['bbox'][2] > b['bbox'][0], f"Invalid width: {b['bbox']}"
            assert b['bbox'][3] > b['bbox'][1], f"Invalid height: {b['bbox']}"

    def test_tolerance_zero_minimal_change(self):
        """With tolerance=0, only pixel snapping should apply."""
        _, jittered = make_toolbar_icons(4, x_start=100, y_top=50,
                                         icon_size_px=32, spacing_px=8,
                                         w=self.W, h=self.H, jitter_px=3)
        result = quantize_boxes(jittered, self.W, self.H, snap_tolerance_px=0)
        for b in result:
            for axis, dim in [(0, self.W), (1, self.H), (2, self.W), (3, self.H)]:
                pixel_val = b['bbox'][axis] * dim
                assert abs(pixel_val - round(pixel_val)) < 1e-9

    def test_large_tolerance_aggressive_grouping(self):
        """Large tolerance should group even somewhat distant boxes."""
        boxes = [
            make_icon_box(0.1, 0.1, 0.1 + 32/1920, 0.1 + 32/1080),
            make_icon_box(0.2, 0.1 + 12/1080, 0.2 + 35/1920, 0.1 + 12/1080 + 35/1080),
        ]
        result = quantize_boxes(boxes, self.W, self.H, snap_tolerance_px=15)
        y1_vals = [b['bbox'][1] for b in result]
        assert len(set(y1_vals)) == 1


# ---------------------------------------------------------------------------
# Comparative metrics: before vs after
# ---------------------------------------------------------------------------

class TestQuantizationMetrics:
    """Compare quantization quality metrics before and after processing."""
    W, H = 1920, 1080

    def _make_scenario(self):
        """Create a realistic toolbar + grid scenario with jitter."""
        gt_toolbar, jit_toolbar = make_toolbar_icons(
            8, x_start=100, y_top=40, icon_size_px=32, spacing_px=8,
            w=self.W, h=self.H, jitter_px=3)

        gt_grid_row1, jit_grid_row1 = make_toolbar_icons(
            5, x_start=200, y_top=300, icon_size_px=48, spacing_px=12,
            w=self.W, h=self.H, jitter_px=4)

        gt_grid_row2, jit_grid_row2 = make_toolbar_icons(
            5, x_start=200, y_top=360, icon_size_px=48, spacing_px=12,
            w=self.W, h=self.H, jitter_px=4)

        gt = gt_toolbar + gt_grid_row1 + gt_grid_row2
        jittered = jit_toolbar + jit_grid_row1 + jit_grid_row2
        return gt, jittered

    def test_edge_variance_decreases(self):
        """Edge alignment variance should decrease after quantization."""
        gt, jittered = self._make_scenario()
        quantized = quantize_boxes(jittered, self.W, self.H, snap_tolerance_px=5)

        for axis, dim in [(1, self.H), (3, self.H)]:
            var_before = edge_variance(jittered[:8], axis, dim)
            var_after = edge_variance(quantized[:8], axis, dim)
            assert var_after <= var_before, \
                f"axis {axis}: variance increased {var_before:.4f} -> {var_after:.4f}"

    def test_size_variance_decreases(self):
        """Size variance should decrease after quantization."""
        gt, jittered = self._make_scenario()
        quantized = quantize_boxes(jittered, self.W, self.H, snap_tolerance_px=5)

        vw_before, vh_before = size_variance(jittered[:8], self.W, self.H)
        vw_after, vh_after = size_variance(quantized[:8], self.W, self.H)
        assert vw_after <= vw_before, \
            f"width variance increased: {vw_before:.4f} -> {vw_after:.4f}"
        assert vh_after <= vh_before, \
            f"height variance increased: {vh_before:.4f} -> {vh_after:.4f}"

    def test_closer_to_ground_truth(self):
        """Quantized boxes should be closer to ground truth than jittered ones."""
        gt, jittered = self._make_scenario()
        quantized = quantize_boxes(jittered, self.W, self.H, snap_tolerance_px=5)

        dist_before = mean_distance_to_gt(jittered, gt, self.W, self.H)
        dist_after = mean_distance_to_gt(quantized, gt, self.W, self.H)
        assert dist_after < dist_before, \
            f"Distance to GT increased: {dist_before:.4f} -> {dist_after:.4f}"

    def test_aspect_ratio_improves(self):
        """Max aspect ratio deviation should decrease for near-square icons."""
        gt, jittered = self._make_scenario()
        quantized = quantize_boxes(jittered, self.W, self.H, snap_tolerance_px=5)

        dev_before = max_aspect_deviation(jittered, self.W, self.H)
        dev_after = max_aspect_deviation(quantized, self.W, self.H)
        assert dev_after <= dev_before, \
            f"Aspect deviation increased: {dev_before:.4f} -> {dev_after:.4f}"

    def test_print_metrics_report(self, capsys):
        """Print a before/after comparison report (informational, always passes)."""
        gt, jittered = self._make_scenario()
        quantized = quantize_boxes(jittered, self.W, self.H, snap_tolerance_px=5)

        print("\n" + "=" * 70)
        print("QUANTIZATION METRICS REPORT")
        print("=" * 70)
        print(f"{'Metric':<35} {'Before':>12} {'After':>12} {'Change':>12}")
        print("-" * 70)

        vb = edge_variance(jittered[:8], 1, self.H)
        va = edge_variance(quantized[:8], 1, self.H)
        print(f"{'Toolbar y1 edge variance (px²)':<35} {vb:>12.4f} {va:>12.4f} {va-vb:>+12.4f}")

        vb = edge_variance(jittered[:8], 3, self.H)
        va = edge_variance(quantized[:8], 3, self.H)
        print(f"{'Toolbar y2 edge variance (px²)':<35} {vb:>12.4f} {va:>12.4f} {va-vb:>+12.4f}")

        wb, hb = size_variance(jittered[:8], self.W, self.H)
        wa, ha = size_variance(quantized[:8], self.W, self.H)
        print(f"{'Toolbar width variance (px²)':<35} {wb:>12.4f} {wa:>12.4f} {wa-wb:>+12.4f}")
        print(f"{'Toolbar height variance (px²)':<35} {hb:>12.4f} {ha:>12.4f} {ha-hb:>+12.4f}")

        db = mean_distance_to_gt(jittered, gt, self.W, self.H)
        da = mean_distance_to_gt(quantized, gt, self.W, self.H)
        print(f"{'Mean L2 distance to GT (px)':<35} {db:>12.4f} {da:>12.4f} {da-db:>+12.4f}")

        ab = max_aspect_deviation(jittered, self.W, self.H)
        aa = max_aspect_deviation(quantized, self.W, self.H)
        print(f"{'Max aspect ratio deviation':<35} {ab:>12.4f} {aa:>12.4f} {aa-ab:>+12.4f}")

        print("=" * 70)
