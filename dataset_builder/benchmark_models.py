import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
from ultralytics import YOLO

from generator import Generator

REAL_DATASET = Path("real_dataset")
SYNTHETIC_TEST_DIR = Path("benchmark_synthetic_test")
SYNTHETIC_TEST_IMAGES = 1000

RESOLUTION = (540, 1200)
TOP_CROP = 60
BOTTOM_CROP = 320

JPEG_QUALITY = (70, 95)
RANDOM_SEED = 17

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def synthetic_test_exists():
    """
    Return True if the synthetic benchmark dataset already exists
    and contains images + labels.
    """

    image_dir = SYNTHETIC_TEST_DIR / "images"
    label_dir = SYNTHETIC_TEST_DIR / "labels"

    if not image_dir.exists() or not label_dir.exists():
        return False

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    return len(images) > 0 and len(labels) > 0


def crop_synthetic(frame, labels):
    """
    Apply the exact same crop used by the training dataset builder.

    Converts generator labels into YOLO pose coordinates.
    """

    width, height = frame.size

    crop_y1 = TOP_CROP
    crop_y2 = height - BOTTOM_CROP
    crop_height = crop_y2 - crop_y1

    frame = frame.crop((0, crop_y1, width, crop_y2))

    output = []

    for label in labels:
        x1 = (label.bbox_x - label.bbox_w / 2) * width

        y1 = (label.bbox_y - label.bbox_h / 2) * height

        x2 = (label.bbox_x + label.bbox_w / 2) * width

        y2 = (label.bbox_y + label.bbox_h / 2) * height

        x1 = max(0, x1)
        y1 = max(crop_y1, y1)
        x2 = min(width, x2)
        y2 = min(crop_y2, y2)

        if x1 >= x2 or y1 >= y2:
            continue

        pivot_x = label.kpt_x * width
        pivot_y = label.kpt_y * height
        keypoint_in_crop = 0 <= pivot_x < width and crop_y1 <= pivot_y < crop_y2
        pivot_y -= crop_y1

        # Convert to cropped YOLO coordinates

        bbox_w = (x2 - x1) / width
        bbox_h = (y2 - y1) / crop_height

        bbox_x = ((x1 + x2) / 2) / width

        bbox_y = ((y1 + y2) / 2 - crop_y1) / crop_height

        pivot_x /= width
        pivot_y /= crop_height

        visibility = (2 if label.kpt_visible else 1) if keypoint_in_crop else 0
        if visibility == 0:
            pivot_x = 0.0
            pivot_y = 0.0

        output.append(
            (
                label.class_id,
                bbox_x,
                bbox_y,
                bbox_w,
                bbox_h,
                pivot_x,
                pivot_y,
                visibility,
            )
        )

    return frame, output


def write_pose_labels(labels, path):
    """Write YOLO pose labels."""

    with open(path, "w") as f:

        for (
            class_id,
            bx,
            by,
            bw,
            bh,
            kx,
            ky,
            visible,
        ) in labels:

            f.write(
                f"{class_id} "
                f"{bx:.6f} "
                f"{by:.6f} "
                f"{bw:.6f} "
                f"{bh:.6f} "
                f"{kx:.6f} "
                f"{ky:.6f} "
                f"{visible}\n"
            )


def create_synthetic_yaml(generator):
    """
    Create a minimal data.yaml for the synthetic benchmark.
    """

    names = generator.asset_manager.class_names

    yaml_path = SYNTHETIC_TEST_DIR / "data.yaml"

    with open(yaml_path, "w") as f:

        f.write(f"path: {SYNTHETIC_TEST_DIR.resolve()}\n")
        f.write("test: images\n\n")
        f.write("kpt_shape: [1, 3]\n\n")
        f.write("names:\n")

        for i, name in enumerate(names):
            f.write(f"  {i}: {name}\n")

    return yaml_path


def load_reference_class_ids():
    path = REAL_DATASET / "data.yaml"
    if not path.exists():
        return None

    config = yaml.safe_load(path.read_text())
    names = config.get("names") if isinstance(config, dict) else None
    if isinstance(names, list):
        return {name: index for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {name: int(index) for index, name in names.items()}
    raise ValueError(f"{path} must contain a names list or mapping")


def generate_synthetic_test():
    """
    Generate the frozen synthetic benchmark dataset.

    This only runs if the dataset does not already exist.
    """

    if synthetic_test_exists():

        image_dir = SYNTHETIC_TEST_DIR / "images"

        existing = [
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print()
        print("=" * 70)
        print("SYNTHETIC TEST SET ALREADY EXISTS")
        print("=" * 70)
        print(f"Found {len(existing)} existing synthetic " "test images.")
        print("Using existing images so the benchmark " "remains reproducible.")

        return

    print()
    print("=" * 70)
    print("GENERATING SYNTHETIC TEST SET")
    print("=" * 70)

    image_dir = SYNTHETIC_TEST_DIR / "images"
    label_dir = SYNTHETIC_TEST_DIR / "labels"

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Generating {SYNTHETIC_TEST_IMAGES} images...")

    print("This will happen only once. " "The resulting dataset will be frozen.")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    generator = Generator(
        RESOLUTION,
        class_name_to_id=load_reference_class_ids(),
    )

    for i in range(SYNTHETIC_TEST_IMAGES):
        frame, labels = generator.random_frame()
        frame, labels = crop_synthetic(
            frame,
            labels,
        )
        name = f"synthetic_test_{i:05d}"
        image_path = image_dir / f"{name}.jpg"
        label_path = label_dir / f"{name}.txt"

        frame.save(
            image_path,
            "JPEG",
            quality=random.randint(*JPEG_QUALITY),
        )

        write_pose_labels(
            labels,
            label_path,
        )

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{SYNTHETIC_TEST_IMAGES}")

    yaml_path = create_synthetic_yaml(generator)

    print()
    print(f"Synthetic test dataset created at:" f"\n  {SYNTHETIC_TEST_DIR.resolve()}")

    print(f"YAML:\n  {yaml_path.resolve()}")


def check_real_test_dataset():
    """Check that the real test dataset exists."""

    yaml_path = REAL_DATASET / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Real dataset YAML not found:\n{yaml_path}")

    config = yaml.safe_load(yaml_path.read_text())
    if not isinstance(config, dict) or "test" not in config:
        raise ValueError(f"{yaml_path} must define a test split for benchmarking")

    dataset_root = Path(config.get("path", REAL_DATASET))
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()

    test_path = Path(config["test"])
    if not test_path.is_absolute():
        test_path = dataset_root / test_path

    image_dir = test_path / "images"
    label_dir = test_path / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"Real test images not found:\n" f"{image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Real test labels not found:\n" f"{label_dir}")

    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise RuntimeError(f"No real test images found in:\n{image_dir}")

    missing_labels = [
        image for image in images if not (label_dir / f"{image.stem}.txt").exists()
    ]
    if missing_labels:
        raise FileNotFoundError(
            f"Missing labels for {len(missing_labels)} real test images "
            f"in:\n{label_dir}"
        )

    print(f"Real test images: {len(images)}")

    return yaml_path


def extract_metrics(results):
    """
    Extract the most useful metrics from an Ultralytics
    validation result.
    """

    metrics = {}

    try:
        metrics["box_precision"] = float(results.box.mp)
        metrics["box_recall"] = float(results.box.mr)
        metrics["box_map50"] = float(results.box.map50)
        metrics["box_map50_95"] = float(results.box.map)
    except Exception:
        metrics["box_precision"] = None
        metrics["box_recall"] = None
        metrics["box_map50"] = None
        metrics["box_map50_95"] = None

    try:
        metrics["pose_precision"] = float(results.pose.mp)
        metrics["pose_recall"] = float(results.pose.mr)
        metrics["pose_map50"] = float(results.pose.map50)
        metrics["pose_map50_95"] = float(results.pose.map)
    except Exception:
        metrics["pose_precision"] = None
        metrics["pose_recall"] = None
        metrics["pose_map50"] = None
        metrics["pose_map50_95"] = None

    return metrics


def evaluate_model(
    model_path,
    data_yaml,
    split,
    output_dir,
):
    """
    Evaluate one model on one dataset.
    """

    print()
    print("-" * 70)
    print(f"Model:   {model_path.name}")
    print(f"Dataset: {data_yaml}")
    print(f"Split:   {split}")
    print("-" * 70)

    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=512,
        batch=16,
        plots=True,
        project=str(output_dir),
        name=(f"{model_path.stem}_{split}"),
        exist_ok=True,
        verbose=True,
    )

    return extract_metrics(results)


def write_csv(results, path):
    """Write benchmark results to CSV."""

    fields = [
        "model",
        "dataset",
        "box_precision",
        "box_recall",
        "box_map50",
        "box_map50_95",
        "pose_precision",
        "pose_recall",
        "pose_map50",
        "pose_map50_95",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)


def print_summary(results):
    """Print a compact benchmark table."""

    print()
    print()
    print("=" * 110)
    print("MODEL BENCHMARK")
    print("=" * 110)

    header = (
        f"{'Model':<28}"
        f"{'Dataset':<12}"
        f"{'Box mAP50':>11}"
        f"{'Box mAP50-95':>14}"
        f"{'Pose mAP50':>12}"
        f"{'Pose mAP50-95':>15}"
    )

    print(header)
    print("-" * 110)

    for row in results:

        model = row["model"]
        dataset = row["dataset"]

        def fmt(value):
            if value is None:
                return "N/A"

            return f"{value:.3f}"

        print(
            f"{model:<28}"
            f"{dataset:<12}"
            f"{fmt(row['box_map50']):>11}"
            f"{fmt(row['box_map50_95']):>14}"
            f"{fmt(row['pose_map50']):>12}"
            f"{fmt(row['pose_map50_95']):>15}"
        )

    print("=" * 110)


def main():
    global SYNTHETIC_TEST_IMAGES

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark YOLO pose models on real "
            "and synthetic Clash Royale test sets."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=("Paths to YOLO .pt model files."),
    )

    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=SYNTHETIC_TEST_IMAGES,
        help=(
            "Number of synthetic test images "
            "to generate if the benchmark "
            "dataset does not already exist."
        ),
    )

    parser.add_argument(
        "--force-generate",
        action="store_true",
        help=(
            "Delete and regenerate the synthetic "
            "test set. Normally you should NOT "
            "use this."
        ),
    )

    args = parser.parse_args()

    SYNTHETIC_TEST_IMAGES = args.synthetic_count
    benchmark_output = Path("benchmark_results")

    benchmark_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.force_generate:
        if SYNTHETIC_TEST_DIR.exists():
            print("WARNING: deleting existing " "synthetic benchmark dataset.")
            import shutil

            shutil.rmtree(SYNTHETIC_TEST_DIR)

    generate_synthetic_test()
    real_yaml = check_real_test_dataset()
    synthetic_yaml = SYNTHETIC_TEST_DIR / "data.yaml"

    if not synthetic_yaml.exists():
        generator = Generator(
            RESOLUTION,
            class_name_to_id=load_reference_class_ids(),
        )

        create_synthetic_yaml(generator)

    models = []

    for model in args.models:
        path = Path(model)
        if not path.exists():
            raise FileNotFoundError(f"Model not found:\n{path}")
        models.append(path)

    all_results = []

    for model_path in models:
        metrics = evaluate_model(
            model_path=model_path,
            data_yaml=real_yaml,
            split="test",
            output_dir=benchmark_output,
        )

        all_results.append(
            {
                "model": model_path.name,
                "dataset": "REAL",
                **metrics,
            }
        )

        metrics = evaluate_model(
            model_path=model_path,
            data_yaml=synthetic_yaml,
            split="test",
            output_dir=benchmark_output,
        )

        all_results.append(
            {
                "model": model_path.name,
                "dataset": "SYNTHETIC",
                **metrics,
            }
        )

    csv_path = benchmark_output / "benchmark_results.csv"

    write_csv(
        all_results,
        csv_path,
    )

    print_summary(all_results)

    print()
    print(f"CSV results saved to:\n" f"  {csv_path.resolve()}")
    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
