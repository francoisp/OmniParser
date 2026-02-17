"""Benchmark quantize_boxes() performance impact."""
import copy
import random
import time
import sys
import os
import types
import ast

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

W, H = 1920, 1080

def make_box(btype, x1, y1, x2, y2, content=None):
    return {'type': btype, 'bbox': [x1, y1, x2, y2],
            'interactivity': btype == 'icon', 'content': content,
            'source': f'box_{btype}'}

def generate_scenario(n_icons, n_text):
    random.seed(42)
    boxes = []
    for i in range(n_text):
        x1 = random.uniform(0, 0.8)
        y1 = random.uniform(0, 0.9)
        boxes.append(make_box('text', x1, y1, x1 + random.uniform(0.05, 0.15),
                              y1 + random.uniform(0.01, 0.03), f'text_{i}'))
    for i in range(n_icons):
        x1 = random.uniform(0, 0.9)
        y1 = random.uniform(0, 0.9)
        size = random.uniform(0.01, 0.04)
        jitter = random.uniform(-0.003, 0.003)
        boxes.append(make_box('icon', x1, y1, x1 + size + jitter, y1 + size - jitter))
    return boxes

def bench(label, boxes, n_runs=1000):
    # Warmup
    for _ in range(10):
        quantize_boxes(boxes, W, H, snap_tolerance_px=5)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        quantize_boxes(boxes, W, H, snap_tolerance_px=5)
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    p99 = times[int(len(times) * 0.99)]
    mean = sum(times) / len(times)
    return label, len(boxes), median, mean, p95, p99

def main():
    scenarios = [
        ("Typical desktop (20 icons, 30 text)", 20, 30),
        ("Dense UI (50 icons, 50 text)", 50, 50),
        ("Heavy screenshot (100 icons, 100 text)", 100, 100),
        ("Extreme (200 icons, 200 text)", 200, 200),
        ("Icons only (50 icons, 0 text)", 50, 0),
        ("Text only (0 icons, 50 text)", 0, 50),
    ]

    print(f"{'Scenario':<45} {'N boxes':>8} {'Median':>10} {'Mean':>10} {'P95':>10} {'P99':>10}")
    print("-" * 95)

    for label, n_icons, n_text in scenarios:
        boxes = generate_scenario(n_icons, n_text)
        lbl, n, median, mean, p95, p99 = bench(label, boxes)
        print(f"{lbl:<45} {n:>8} {median*1000:>9.3f}ms {mean*1000:>9.3f}ms {p95*1000:>9.3f}ms {p99*1000:>9.3f}ms")

    # Compare against a no-op baseline (just deepcopy, the minimum cost)
    print("\n--- Baseline comparison (typical desktop scenario) ---")
    boxes = generate_scenario(20, 30)
    n_runs = 1000

    # deepcopy only
    times_copy = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        copy.deepcopy(boxes)
        times_copy.append(time.perf_counter() - t0)
    times_copy.sort()

    # full quantize
    times_quant = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        quantize_boxes(boxes, W, H, snap_tolerance_px=5)
        times_quant.append(time.perf_counter() - t0)
    times_quant.sort()

    copy_med = times_copy[len(times_copy) // 2] * 1000
    quant_med = times_quant[len(times_quant) // 2] * 1000
    print(f"  deepcopy only:    {copy_med:.3f}ms")
    print(f"  quantize_boxes:   {quant_med:.3f}ms")
    print(f"  overhead:         {quant_med - copy_med:.3f}ms")
    print(f"\nFor reference, YOLO inference typically takes 50-500ms, OCR 100-2000ms.")

if __name__ == '__main__':
    main()
