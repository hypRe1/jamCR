import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from generator import Generator


OUTPUT_DIR = Path("dataset")
REAL_DATASET_DIR = Path("real_dataset")
NEGATIVE_DIR = Path("negatives")

RESOLUTION = (540, 1200)

TOP_CROP = 60
BOTTOM_CROP = 320

SYNTHETIC_TRAIN = 10000
SYNTHETIC_VAL = 1000
SYNTHETIC_TEST = 0

REAL_TEST = 0

NUM_NEGATIVES = 200

RANDOM_SEED = 42

JPEG_QUALITY = (70, 95)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def reset_dataset():
    """Delete and recreate the output dataset."""

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ("train", "val", "test"):
        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def crop_synthetic(frame, labels):
    """
    Crop the synthetic frame and convert labels from the
    generator's full-frame coordinates to cropped YOLO pose
    coordinates.

    Each synthetic label contains:

        class
        bbox
        pivot
        pivot visibility
    """

    width, height = frame.size

    crop_y1 = TOP_CROP
    crop_y2 = height - BOTTOM_CROP
    crop_height = crop_y2 - crop_y1

    frame = frame.crop(
        (0, crop_y1, width, crop_y2)
    )

    output = []

    for label in labels:

        x1 = (
            label.bbox_x
            - label.bbox_w / 2
        ) * width

        y1 = (
            label.bbox_y
            - label.bbox_h / 2
        ) * height

        x2 = (
            label.bbox_x
            + label.bbox_w / 2
        ) * width

        y2 = (
            label.bbox_y
            + label.bbox_h / 2
        ) * height

        x1 = max(0, x1)
        y1 = max(crop_y1, y1)
        x2 = min(width, x2)
        y2 = min(crop_y2, y2)

        if x1 >= x2 or y1 >= y2:
            continue

        pivot_x = label.kpt_x * width
        pivot_y = label.kpt_y * height
        keypoint_in_crop = (
            0 <= pivot_x < width
            and crop_y1 <= pivot_y < crop_y2
        )
        pivot_y -= crop_y1

        bbox_w = (x2 - x1) / width
        bbox_h = (y2 - y1) / crop_height

        bbox_x = ((x1 + x2) / 2) / width
        bbox_y = (
            ((y1 + y2) / 2 - crop_y1)
            / crop_height
        )

        pivot_x /= width
        pivot_y /= crop_height

        visibility = (
            2 if label.kpt_visible else 1
        ) if keypoint_in_crop else 0
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


def find_real_test_images():
    """
    Collect all real images from the existing train and val
    directories, shuffle them, and select REAL_TEST images.

    The real images are used ONLY for testing in this dataset.
    """

    images = []

    for split in ("train", "val"):

        image_dir = (
            REAL_DATASET_DIR
            / "images"
            / split
        )

        label_dir = (
            REAL_DATASET_DIR
            / "labels"
            / split
        )

        if not image_dir.exists():
            continue

        for image in image_dir.iterdir():

            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label = (
                label_dir
                / f"{image.stem}.txt"
            )

            if not label.exists():
                raise FileNotFoundError(
                    f"Missing label:\n{label}"
                )

            images.append((image, label))

    if not images:
        raise RuntimeError(
            "No real images found in "
            f"{REAL_DATASET_DIR}/images/train "
            f"or {REAL_DATASET_DIR}/images/val."
        )

    print(
        f"Found {len(images)} real images "
        f"across train/val."
    )

    # Deterministic shuffle so the test set is reproducible.
    random.Random(
        RANDOM_SEED
    ).shuffle(images)

    if len(images) < REAL_TEST:
        raise RuntimeError(
            f"Need {REAL_TEST} real images, "
            f"but only found {len(images)}."
        )

    return images[:REAL_TEST]


def convert_real_label_to_pose(src, dst):
    """
    Convert a real detection label into a pose label.

    The real test images do NOT contain pivot supervision.

    Therefore:

        keypoint_x = 0
        keypoint_y = 0
        visibility = 0

    This means the keypoint is not labelled.
    """

    with open(src) as f:
        lines = f.readlines()

    with open(dst, "w") as f:

        for line in lines:

            parts = line.split()

            if len(parts) < 5:
                continue

            class_id = parts[0]
            bbox = parts[1:5]

            f.write(
                f"{class_id} "
                f"{' '.join(bbox)} "
                f"0 0 0\n"
            )


def copy_real_test(images):
    """
    Copy real test images into the test split.

    These images are NEVER used for training or validation.
    """

    for i, (image, label) in enumerate(images):

        output_name = (
            f"real_test_{i:05d}"
        )

        shutil.copy2(
            image,
            OUTPUT_DIR
            / "images"
            / "test"
            / f"{output_name}{image.suffix.lower()}",
        )

        convert_real_label_to_pose(
            label,
            OUTPUT_DIR
            / "labels"
            / "test"
            / f"{output_name}.txt",
        )


def generate_synthetic(generator, split, count):
    """Generate synthetic pose data."""

    print(
        f"Generating {count} synthetic "
        f"{split} images..."
    )

    for i in range(count):

        frame, labels = (
            generator.random_frame()
        )

        frame, labels = crop_synthetic(
            frame,
            labels,
        )

        name = (
            f"synthetic_{split}_{i:05d}"
        )

        image_path = (
            OUTPUT_DIR
            / "images"
            / split
            / f"{name}.jpg"
        )

        label_path = (
            OUTPUT_DIR
            / "labels"
            / split
            / f"{name}.txt"
        )

        frame.save(
            image_path,
            "JPEG",
            quality=random.randint(
                *JPEG_QUALITY
            ),
        )

        write_pose_labels(
            labels,
            label_path,
        )

        if (i + 1) % 100 == 0:
            print(
                f"  {i + 1}/{count}"
            )


def add_negatives():
    """
    Add empty-label negative images to TRAINING ONLY.
    """

    if not NEGATIVE_DIR.exists():
        print("No negatives directory found.")
        return 0

    images = [
        p
        for p in NEGATIVE_DIR.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    random.Random(
        RANDOM_SEED
    ).shuffle(images)

    images = images[:NUM_NEGATIVES]

    for i, image in enumerate(images):

        frame = Image.open(
            image
        ).convert("RGB")

        # Match synthetic preprocessing
        if frame.size != RESOLUTION:
            frame = frame.resize(
                RESOLUTION,
                Image.Resampling.LANCZOS,
            )

        frame = frame.crop(
            (
                0,
                TOP_CROP,
                frame.width,
                frame.height - BOTTOM_CROP,
            )
        )

        name = f"negative_{i:05d}"

        frame.save(
            OUTPUT_DIR
            / "images"
            / "train"
            / f"{name}.jpg",
            "JPEG",
            quality=random.randint(
                *JPEG_QUALITY
            ),
        )

        # Empty YOLO label
        (
            OUTPUT_DIR
            / "labels"
            / "train"
            / f"{name}.txt"
        ).touch()

    return len(images)


def create_yaml(generator):
    """Create YOLO pose data.yaml."""

    names = generator.asset_manager.class_names

    path = OUTPUT_DIR / "data.yaml"

    with open(path, "w") as f:

        f.write(
            f"path: {OUTPUT_DIR.resolve()}\n"
        )

        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")

        f.write("kpt_shape: [1, 3]\n\n")
        f.write("flip_idx: [0]\n\n")

        f.write("names:\n")

        for i, name in enumerate(names):
            f.write(
                f"  {i}: {name}\n"
            )

    return path


def load_reference_class_ids():
    path = REAL_DATASET_DIR / "data.yaml"
    if not path.exists():
        return None

    config = yaml.safe_load(path.read_text())
    names = config.get("names", {})
    if isinstance(names, list):
        return {name: index for index, name in enumerate(names)}
    return {
        name: int(index)
        for index, name in names.items()
    }


def main(
    output_dir: Path | None = None,
    synthetic_train: int | None = None,
    synthetic_val: int | None = None,
    synthetic_test: int | None = None,
    num_negatives: int | None = None,
    asset_dir: Path | None = None,
):
    global OUTPUT_DIR, SYNTHETIC_TRAIN, SYNTHETIC_VAL
    global SYNTHETIC_TEST, NUM_NEGATIVES

    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir)
    if synthetic_train is not None:
        SYNTHETIC_TRAIN = synthetic_train
    if synthetic_val is not None:
        SYNTHETIC_VAL = synthetic_val
    if synthetic_test is not None:
        SYNTHETIC_TEST = synthetic_test
    if num_negatives is not None:
        NUM_NEGATIVES = num_negatives

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("BUILDING SYNTHETIC-ONLY YOLO POSE DATASET")
    print("=" * 60)

    print()
    print("TRAIN")
    print(f"  Synthetic: {SYNTHETIC_TRAIN}")
    print(f"  Negatives: {NUM_NEGATIVES}")

    print()
    print("VAL")
    print(f"  Synthetic: {SYNTHETIC_VAL}")

    print()
    print("TEST")
    print(f"  Synthetic: {SYNTHETIC_TEST}")
    print(f"  Real:      {REAL_TEST}")

    print()
    print("Initializing generator...")

    generator = Generator(
        RESOLUTION,
        class_name_to_id=load_reference_class_ids(),
        asset_dir=asset_dir,
    )

    reset_dataset()

    print()
    print("=" * 60)
    print("SYNTHETIC DATA")
    print("=" * 60)

    generate_synthetic(
        generator,
        "train",
        SYNTHETIC_TRAIN,
    )

    generate_synthetic(
        generator,
        "val",
        SYNTHETIC_VAL,
    )

    generate_synthetic(
        generator,
        "test",
        SYNTHETIC_TEST,
    )

    if REAL_TEST:
        copy_real_test(find_real_test_images())

    print()
    print("=" * 60)
    print("NEGATIVES")
    print("=" * 60)

    negatives = add_negatives()

    print(
        f"Added {negatives} negative images "
        f"to training."
    )

    yaml_path = create_yaml(
        generator
    )

    print()
    print("=" * 60)
    print("DATASET COMPLETE")
    print("=" * 60)

    print()
    print("TRAIN")
    print(
        f"  Synthetic: {SYNTHETIC_TRAIN}"
    )
    print(
        f"  Negatives: {negatives}"
    )
    print(
        f"  Total:     "
        f"{SYNTHETIC_TRAIN + negatives}"
    )

    print()
    print("VAL")
    print(
        f"  Synthetic: {SYNTHETIC_VAL}"
    )

    print()
    print("TEST")
    print(
        f"  Synthetic: {SYNTHETIC_TEST}"
    )
    print(
        f"  Real:      {REAL_TEST}"
    )
    print(
        f"  Total:     "
        f"{SYNTHETIC_TEST + REAL_TEST}"
    )

    print()
    print("LABELS")
    print("  Synthetic: bbox + pivot")
    print("  Negatives: empty")
    print("  Real test: bbox only, pivot unavailable")

    print()
    print(f"Dataset YAML: {yaml_path}")
    print("Done!")


if __name__ == "__main__":
    main()
