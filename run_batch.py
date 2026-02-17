#!/usr/bin/env python3
"""Run OmniParser on raw screenshots and save annotations for comparison.

Usage:
    python run_batch.py [--n 5] [--snap] [--blank-filter]

Processes the first N screenshots from data/raw_screenshots/ and saves
annotated images + JSON to data/new_annotations/.
"""
import argparse
import base64
import io
import json
import os
import sys
import time

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description='Batch process screenshots with OmniParser')
    parser.add_argument('--n', type=int, default=5, help='Number of screenshots to process')
    parser.add_argument('--snap', action='store_true', help='Enable box snapping/quantization')
    parser.add_argument('--snap-tolerance', type=int, default=5, help='Snap tolerance in pixels')
    parser.add_argument('--blank-filter', action='store_true', help='Enable blank icon filter')
    parser.add_argument('--blank-threshold', type=float, default=5.0, help='Blank stddev threshold')
    args = parser.parse_args()

    raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw_screenshots')
    out_dir = os.path.join(os.path.dirname(__file__), 'data', 'new_annotations')
    os.makedirs(out_dir, exist_ok=True)

    screenshots = sorted([f for f in os.listdir(raw_dir) if f.endswith('.png')])[:args.n]
    print(f"Processing {len(screenshots)} screenshots...")
    print(f"  snap_enabled={args.snap}, blank_filter={args.blank_filter}")
    print(f"  Output: {out_dir}")
    print()

    # Import and load models (heavy — do once)
    from util.utils import (
        check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
    )

    print("Loading YOLO model...")
    yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
    print("Loading caption model...")
    caption_model_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path="weights/icon_caption_florence"
    )
    print("Models loaded.\n")

    for i, fname in enumerate(screenshots):
        print(f"[{i+1}/{len(screenshots)}] {fname}")
        t0 = time.time()

        img_path = os.path.join(raw_dir, fname)
        image = Image.open(img_path)
        w, h = image.size

        box_overlay_ratio = max(w, h) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # OCR
        (text, ocr_bbox), _ = check_ocr_box(
            image, display_img=False, output_bb_format='xyxy',
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=True
        )

        # Full pipeline
        dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image, yolo_model,
            BOX_TRESHOLD=0.05,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=0.1,
            imgsz=640,
            snap_enabled=args.snap,
            snap_tolerance_px=args.snap_tolerance,
            blank_filter_enabled=args.blank_filter,
            blank_stddev_threshold=args.blank_threshold,
        )

        elapsed = time.time() - t0

        # Save annotated image
        annotated_img = Image.open(io.BytesIO(base64.b64decode(dino_labeled_img)))
        stem = os.path.splitext(fname)[0]
        annotated_img.save(os.path.join(out_dir, f"{stem}_annotated.png"))

        # Save JSON
        annotation = {
            'image': fname,
            'image_size': {'width': w, 'height': h},
            'latency': elapsed,
            'elements_found': len(parsed_content_list),
            'settings': {
                'snap_enabled': args.snap,
                'snap_tolerance_px': args.snap_tolerance,
                'blank_filter_enabled': args.blank_filter,
                'blank_stddev_threshold': args.blank_threshold,
            },
            'parsed_content_list': parsed_content_list,
        }
        with open(os.path.join(out_dir, f"{stem}.json"), 'w') as f:
            json.dump(annotation, f, indent=2)

        n_icons = sum(1 for e in parsed_content_list if e['type'] == 'icon')
        n_text = sum(1 for e in parsed_content_list if e['type'] == 'text')
        print(f"  -> {len(parsed_content_list)} elements ({n_icons} icons, {n_text} text) in {elapsed:.1f}s")

    print(f"\nDone. Results in {out_dir}/")


if __name__ == '__main__':
    main()
