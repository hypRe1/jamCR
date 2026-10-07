from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DATASET_BUILDER = ROOT / "dataset_builder"
if str(DATASET_BUILDER) not in sys.path:
    sys.path.insert(0, str(DATASET_BUILDER))

from generator import Generator

RESOLUTION = (540, 1200)


def class_name_for_id(generator: Generator, class_id: int) -> str:
    names = generator.asset_manager.class_names
    if 0 <= class_id < len(names):
        return names[class_id]
    return f"class_{class_id}"


def draw_label_overlay(frame: Image.Image, labels, generator: Generator, output_path: Path) -> None:
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    width, height = frame.size

    for label in labels:
        bbox_x = label.bbox_x * width
        bbox_y = label.bbox_y * height
        bbox_w = label.bbox_w * width
        bbox_h = label.bbox_h * height

        x1 = bbox_x - bbox_w / 2
        y1 = bbox_y - bbox_h / 2
        x2 = bbox_x + bbox_w / 2
        y2 = bbox_y + bbox_h / 2

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(0, 255, 0),
            width=2,
        )

        kx = label.kpt_x * width
        ky = label.kpt_y * height

        draw.ellipse(
            [kx - 5, ky - 5, kx + 5, ky + 5],
            fill=(255, 0, 0),
            outline=(255, 255, 255),
            width=2,
        )

        name = class_name_for_id(generator, label.class_id)
        label_text = f"{name} | occ={label.occlusion_ratio:.2f}"

        text_x = max(8, x1)
        text_y = max(8, y1 - 18)

        draw.text(
            (text_x, text_y),
            label_text,
            fill=(255, 255, 255),
            stroke_fill=(0, 0, 0),
            stroke_width=1,
            font=font,
        )

    frame.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one rendered synthetic Clash Royale sample with visible labels, boxes and keypoints."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("debug_render_sample.png"),
        help="Path to write the debug image to.",
    )
    args = parser.parse_args()

    generator = Generator(RESOLUTION)
    frame, labels = generator.random_frame()

    draw_label_overlay(frame, labels, generator, args.out)
    print(f"Saved debug render to: {args.out.resolve()}")
    print(f"Generated {len(labels)} labels")


if __name__ == "__main__":
    main()
