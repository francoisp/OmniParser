"""Visual comparison of bounding boxes before and after quantization.

Generates a side-by-side PNG showing jittered (before) vs quantized (after)
boxes on a synthetic UI-like background.
"""
import copy
import random
import sys
import os
import types
import ast

# ---------------------------------------------------------------------------
# Load quantize functions without heavy ML imports
# ---------------------------------------------------------------------------
def _load_quantize_functions():
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
quantize_boxes = _qmod.quantize_boxes

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080

def make_icon_box(x1, y1, x2, y2):
    return {'type': 'icon', 'bbox': [x1, y1, x2, y2],
            'interactivity': True, 'content': None, 'source': 'box_yolo_content_yolo'}

def make_text_box(x1, y1, x2, y2, content='text'):
    return {'type': 'text', 'bbox': [x1, y1, x2, y2],
            'interactivity': False, 'content': content, 'source': 'box_ocr_content_ocr'}

def add_jitter(val, max_jitter_px, dim):
    jitter = random.uniform(-max_jitter_px, max_jitter_px) / dim
    return max(0.0, min(1.0, val + jitter))

def make_toolbar_icons(n, x_start, y_top, icon_size_px, spacing_px, jitter_px=3):
    gt, jittered = [], []
    for i in range(n):
        x1 = (x_start + i * (icon_size_px + spacing_px)) / W
        y1 = y_top / H
        x2 = (x_start + i * (icon_size_px + spacing_px) + icon_size_px) / W
        y2 = (y_top + icon_size_px) / H
        gt.append(make_icon_box(x1, y1, x2, y2))
        jittered.append(make_icon_box(
            add_jitter(x1, jitter_px, W), add_jitter(y1, jitter_px, H),
            add_jitter(x2, jitter_px, W), add_jitter(y2, jitter_px, H)))
    return gt, jittered

def draw_boxes(draw, boxes, color, label_prefix="", offset_x=0):
    for i, b in enumerate(boxes):
        bbox = b['bbox']
        x1 = int(bbox[0] * W) + offset_x
        y1 = int(bbox[1] * H)
        x2 = int(bbox[2] * W) + offset_x
        y2 = int(bbox[3] * H)
        outline_color = color if b['type'] == 'icon' else '#888888'
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=2)
        draw.text((x1 + 2, y1 + 2), f"{label_prefix}{i}", fill=outline_color)

def draw_grid_lines(draw, boxes, color, offset_x=0):
    """Draw faint horizontal/vertical guide lines through aligned edges."""
    icon_boxes = [b for b in boxes if b['type'] == 'icon']
    if not icon_boxes:
        return
    # Find groups of identical y1 and y2 values
    from collections import Counter
    for axis, dim in [(1, H), (3, H)]:
        vals = [round(b['bbox'][axis] * dim) for b in icon_boxes]
        counts = Counter(vals)
        for val, count in counts.items():
            if count >= 2:
                y = val
                draw.line([(offset_x, y), (offset_x + W, y)],
                         fill=color, width=1)
    for axis, dim in [(0, W), (2, W)]:
        vals = [round(b['bbox'][axis] * dim) for b in icon_boxes]
        counts = Counter(vals)
        for val, count in counts.items():
            if count >= 2:
                x = val + offset_x
                draw.line([(x, 0), (x, H)],
                         fill=color, width=1)


def main():
    random.seed(42)

    # Build a realistic scenario with multiple UI regions
    all_gt, all_jittered = [], []

    # Text boxes (OCR) - these should pass through unchanged
    text_boxes = [
        make_text_box(50/W, 10/H, 200/W, 30/H, 'File'),
        make_text_box(210/W, 10/H, 340/W, 30/H, 'Edit'),
        make_text_box(350/W, 10/H, 500/W, 30/H, 'View'),
    ]

    # Toolbar row 1: small icons
    gt1, jit1 = make_toolbar_icons(10, x_start=50, y_top=45, icon_size_px=28, spacing_px=6, jitter_px=3)
    # Toolbar row 2: medium icons
    gt2, jit2 = make_toolbar_icons(8, x_start=50, y_top=85, icon_size_px=36, spacing_px=8, jitter_px=4)
    # Side panel grid: icon grid (2 rows x 6 cols)
    gt3, jit3 = make_toolbar_icons(6, x_start=1400, y_top=150, icon_size_px=48, spacing_px=12, jitter_px=4)
    gt4, jit4 = make_toolbar_icons(6, x_start=1400, y_top=220, icon_size_px=48, spacing_px=12, jitter_px=4)
    gt5, jit5 = make_toolbar_icons(6, x_start=1400, y_top=290, icon_size_px=48, spacing_px=12, jitter_px=4)
    # Bottom status bar icons
    gt6, jit6 = make_toolbar_icons(5, x_start=800, y_top=1040, icon_size_px=24, spacing_px=10, jitter_px=2)

    all_gt = gt1 + gt2 + gt3 + gt4 + gt5 + gt6
    all_jittered = text_boxes + jit1 + jit2 + jit3 + jit4 + jit5 + jit6
    all_gt_with_text = text_boxes + gt1 + gt2 + gt3 + gt4 + gt5 + gt6

    # Quantize
    quantized = quantize_boxes(all_jittered, W, H, snap_tolerance_px=5)

    # Create side-by-side image
    canvas = Image.new('RGB', (W * 2 + 20, H + 60), '#1e1e1e')
    draw = ImageDraw.Draw(canvas)

    # Labels
    draw.text((W // 2 - 80, 5), "BEFORE (jittered YOLO output)", fill='#ff6666')
    draw.text((W + 20 + W // 2 - 80, 5), "AFTER (quantized)", fill='#66ff66')

    # Draw guide lines (faint) to show alignment
    draw_grid_lines(draw, all_jittered, '#331111', offset_x=0)
    draw_grid_lines(draw, quantized, '#113311', offset_x=W + 20)

    # Draw ground truth boxes (very faint) for reference
    for b in all_gt_with_text:
        bbox = b['bbox']
        for ox in [0, W + 20]:
            x1 = int(bbox[0] * W) + ox
            y1 = int(bbox[1] * H) + 30
            x2 = int(bbox[2] * W) + ox
            y2 = int(bbox[3] * H) + 30
            draw.rectangle([x1, y1, x2, y2], outline='#333333', width=1)

    # Shift y for labels area
    def shifted_boxes(boxes, dy=30):
        result = []
        for b in boxes:
            nb = copy.deepcopy(b)
            nb['bbox'][1] += dy / H
            nb['bbox'][3] += dy / H
            result.append(nb)
        return result

    jit_shifted = shifted_boxes(all_jittered)
    quant_shifted = shifted_boxes(quantized)

    # Draw before boxes (left)
    draw_boxes(draw, jit_shifted, '#ff4444', offset_x=0)
    # Draw after boxes (right)
    draw_boxes(draw, quant_shifted, '#44ff44', offset_x=W + 20)

    # Add stats at the bottom
    from collections import Counter
    def count_unique_edges(boxes, axis, dim):
        icon_boxes = [b for b in boxes if b['type'] == 'icon']
        vals = [round(b['bbox'][axis] * dim, 1) for b in icon_boxes]
        return len(set(vals))

    y_text = H + 35
    stats = [
        f"Unique y1 edges: {count_unique_edges(all_jittered, 1, H)} -> {count_unique_edges(quantized, 1, H)}",
        f"   Unique y2 edges: {count_unique_edges(all_jittered, 3, H)} -> {count_unique_edges(quantized, 3, H)}",
        f"   Icon boxes: {sum(1 for b in all_jittered if b['type']=='icon')}",
        f"   Text boxes (unchanged): {sum(1 for b in all_jittered if b['type']=='text')}",
    ]
    draw.text((10, y_text), "  ".join(stats), fill='#aaaaaa')

    out_path = os.path.join(os.path.dirname(__file__), 'quantize_comparison.png')
    canvas.save(out_path)
    print(f"Saved comparison image to: {out_path}")
    return out_path


if __name__ == '__main__':
    main()
