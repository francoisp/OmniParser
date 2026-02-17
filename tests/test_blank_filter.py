"""Tests for the blank icon filter.

Tests filter_blank_icons() with synthetic images to verify it correctly
drops solid/near-solid regions while keeping real icon-like content.
Also validates against the actual annotation data if available.
"""
import copy
import os
import sys
import types
import ast
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load filter function without heavy ML imports
# ---------------------------------------------------------------------------
def _load_filter_function():
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'util', 'utils.py')
    with open(utils_path) as f:
        source = f.read()
    tree = ast.parse(source)
    target_funcs = {'filter_blank_icons'}
    func_sources = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_funcs:
            func_sources.append(ast.get_source_segment(source, node))
    module = types.ModuleType('filter_funcs')
    module.__dict__['np'] = np
    for src in func_sources:
        exec(compile(src, '<filter>', 'exec'), module.__dict__)
    return module

_fmod = _load_filter_function()
filter_blank_icons = _fmod.filter_blank_icons


W, H = 1920, 1080

def make_icon(x1, y1, x2, y2):
    return {'type': 'icon', 'bbox': [x1, y1, x2, y2],
            'interactivity': True, 'content': None}


class TestFilterBlankIcons:

    def _solid_image(self, color=(255, 255, 255)):
        """Create a solid-color image."""
        return np.full((H, W, 3), color, dtype=np.uint8)

    def _image_with_icon(self, bg_color=(240, 240, 240)):
        """Create image with a solid background and a drawn 'icon' region."""
        img = np.full((H, W, 3), bg_color, dtype=np.uint8)
        # Draw a colored pattern in the icon region (100:132, 100:132)
        img[100:132, 100:132, 0] = 200  # red
        img[100:132, 100:132, 1] = 50   # green
        img[100:132, 100:132, 2] = 50   # blue
        # Add some noise/edges
        img[110:120, 110:120] = [0, 0, 0]
        return img

    def test_drops_solid_white(self):
        img = self._solid_image((255, 255, 255))
        boxes = [make_icon(0.05, 0.05, 0.1, 0.1)]
        result = filter_blank_icons(boxes, img, W, H, stddev_threshold=5.0)
        assert len(result) == 0

    def test_drops_solid_black(self):
        img = self._solid_image((0, 0, 0))
        boxes = [make_icon(0.05, 0.05, 0.1, 0.1)]
        result = filter_blank_icons(boxes, img, W, H, stddev_threshold=5.0)
        assert len(result) == 0

    def test_drops_near_solid_gray(self):
        img = self._solid_image((200, 200, 200))
        # Add tiny noise (stddev ~1)
        np.random.seed(42)
        noise = np.random.randint(-2, 3, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        boxes = [make_icon(0.05, 0.05, 0.1, 0.1)]
        result = filter_blank_icons(boxes, img, W, H, stddev_threshold=5.0)
        assert len(result) == 0

    def test_keeps_real_icon(self):
        img = self._image_with_icon()
        # Box covering the drawn icon region
        boxes = [make_icon(100/W, 100/H, 132/W, 132/H)]
        result = filter_blank_icons(boxes, img, W, H, stddev_threshold=5.0)
        assert len(result) == 1

    def test_mixed_keeps_only_real(self):
        img = self._image_with_icon(bg_color=(250, 250, 250))
        boxes = [
            make_icon(100/W, 100/H, 132/W, 132/H),   # real icon
            make_icon(500/W, 500/H, 550/W, 550/H),    # blank area
            make_icon(800/W, 800/H, 850/W, 850/H),    # blank area
        ]
        result = filter_blank_icons(boxes, img, W, H, stddev_threshold=5.0)
        assert len(result) == 1
        # Check it kept the right one
        assert abs(result[0]['bbox'][0] - 100/W) < 0.001

    def test_empty_input(self):
        img = self._solid_image()
        result = filter_blank_icons([], img, W, H)
        assert result == []

    def test_threshold_zero_keeps_all_nonzero(self):
        """With threshold=0, only perfectly uniform regions are dropped."""
        img = self._solid_image((200, 200, 200))
        np.random.seed(42)
        noise = np.random.randint(-1, 2, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        boxes = [make_icon(0.05, 0.05, 0.1, 0.1)]
        result = filter_blank_icons(boxes, img, W, H, stddev_threshold=0.0)
        # Should keep it since noise makes std > 0
        assert len(result) == 1

    def test_does_not_mutate_input(self):
        img = self._solid_image()
        boxes = [make_icon(0.05, 0.05, 0.1, 0.1)]
        original = copy.deepcopy(boxes)
        filter_blank_icons(boxes, img, W, H)
        assert boxes == original

    def test_invalid_box_skipped(self):
        """Boxes with zero area should be silently skipped."""
        img = self._solid_image()
        boxes = [make_icon(0.1, 0.1, 0.1, 0.1)]  # zero area
        result = filter_blank_icons(boxes, img, W, H)
        assert len(result) == 0


class TestFilterOnRealData:
    """Validate filter against actual annotation data if available."""

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    ANN_DIR = os.path.join(DATA_DIR, 'annotations')
    RAW_DIR = os.path.join(DATA_DIR, 'raw_screenshots')

    @pytest.fixture(autouse=True)
    def check_data(self):
        if not os.path.exists(self.ANN_DIR) or not os.path.exists(self.RAW_DIR):
            pytest.skip("No annotation data available")
        ann_files = [f for f in os.listdir(self.ANN_DIR) if f.endswith('.json')]
        if not ann_files:
            pytest.skip("No annotation files found")

    def _load_all(self):
        import json
        from PIL import Image
        ann_files = sorted([f for f in os.listdir(self.ANN_DIR) if f.endswith('.json')])
        for fname in ann_files:
            with open(os.path.join(self.ANN_DIR, fname)) as fh:
                d = json.load(fh)
            img_path = os.path.join(self.RAW_DIR, d['image'])
            if not os.path.exists(img_path):
                continue
            img = np.array(Image.open(img_path).convert('RGB'))
            yield d, img

    def test_filter_removes_known_blanks(self):
        """Filter should remove regions with known blank captions."""
        blank_words = {'white', 'black', 'blank', 'empty', 'm0,0l',
                       'a blank space', 'keystone'}
        total_blanks = 0
        caught = 0

        for d, img in self._load_all():
            w, h = d['image_size']['width'], d['image_size']['height']
            icons = [e for e in d['parsed_content_list'] if e['type'] == 'icon']
            for elem in icons:
                content = (elem['content'] or '').strip().lower()
                if not any(kw in content for kw in blank_words):
                    continue
                # This is a known blank - check if filter would catch it
                bbox = elem['bbox']
                norm_bbox = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
                icon_elem = [{'type': 'icon', 'bbox': norm_bbox}]
                result = filter_blank_icons(icon_elem, img, w, h, stddev_threshold=5.0)
                total_blanks += 1
                if len(result) == 0:
                    caught += 1

        catch_rate = caught / total_blanks if total_blanks > 0 else 0
        print(f"\nBlank filter catch rate: {caught}/{total_blanks} ({100*catch_rate:.1f}%)")
        # Should catch at least 20% of known blanks at conservative threshold
        assert catch_rate > 0.20, f"Only caught {100*catch_rate:.1f}%"

    def test_filter_preserves_most_real_icons(self):
        """Filter should keep >95% of non-blank icons."""
        blank_words = {'white', 'black', 'blank', 'empty', 'm0,0l',
                       'a blank space', 'keystone'}
        total_real = 0
        kept = 0

        for d, img in self._load_all():
            w, h = d['image_size']['width'], d['image_size']['height']
            icons = [e for e in d['parsed_content_list'] if e['type'] == 'icon']
            for elem in icons:
                content = (elem['content'] or '').strip().lower()
                if any(kw in content for kw in blank_words):
                    continue
                bbox = elem['bbox']
                norm_bbox = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
                icon_elem = [{'type': 'icon', 'bbox': norm_bbox}]
                result = filter_blank_icons(icon_elem, img, w, h, stddev_threshold=5.0)
                total_real += 1
                if len(result) == 1:
                    kept += 1

        keep_rate = kept / total_real if total_real > 0 else 0
        print(f"\nReal icon keep rate: {kept}/{total_real} ({100*keep_rate:.1f}%)")
        assert keep_rate > 0.95, f"Only kept {100*keep_rate:.1f}%"

    def test_print_filter_report(self, capsys):
        """Print per-threshold report across all data."""
        import json
        from PIL import Image

        blank_words = {'white', 'black', 'blank', 'empty', 'm0,0l',
                       'a blank space', 'keystone'}

        results_by_threshold = {t: {'blank_caught': 0, 'real_killed': 0}
                                for t in [3, 5, 8, 10, 15]}
        total_blank = 0
        total_real = 0

        for d, img in self._load_all():
            w, h = d['image_size']['width'], d['image_size']['height']
            icons = [e for e in d['parsed_content_list'] if e['type'] == 'icon']
            for elem in icons:
                content = (elem['content'] or '').strip().lower()
                is_blank = any(kw in content for kw in blank_words)
                bbox = elem['bbox']
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(w, int(bbox[2])), min(h, int(bbox[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                std = float(crop.astype(np.float32).std(axis=(0,1)).mean())

                if is_blank:
                    total_blank += 1
                    for t in results_by_threshold:
                        if std < t:
                            results_by_threshold[t]['blank_caught'] += 1
                else:
                    total_real += 1
                    for t in results_by_threshold:
                        if std < t:
                            results_by_threshold[t]['real_killed'] += 1

        print(f"\n{'='*65}")
        print("BLANK FILTER ANALYSIS ON REAL DATA")
        print(f"{'='*65}")
        print(f"Total icons: {total_blank + total_real} "
              f"(blank: {total_blank}, real: {total_real})")
        print(f"\n{'Threshold':>10} {'Blanks caught':>15} {'Real killed':>15} {'Net gain':>10}")
        print("-" * 55)
        for t in sorted(results_by_threshold.keys()):
            r = results_by_threshold[t]
            bc = r['blank_caught']
            rk = r['real_killed']
            print(f"{t:>10} {bc:>8}/{total_blank} ({100*bc/max(1,total_blank):>4.1f}%)"
                  f" {rk:>7}/{total_real} ({100*rk/max(1,total_real):>4.1f}%)"
                  f" {bc-rk:>+10}")
        print(f"{'='*65}")
