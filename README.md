# jamCR

jamCR is a vision project for Clash Royale. It builds a synthetic dataset
from game sprites and arena assets, then uses that data to train a YOLO pose
model that detects both the bounding boxes and exact positions of troops and
buildings.

## Examples

Synthetic scene with generated labels:k

![Synthetic scene with bounding boxes and world-anchor keypoints](docs/images/debug_render_sample.png)

Model predictions on a real game image:

![YOLO pose predictions on a real game image](docs/images/prediction.png)


## Setup

The project requires Python 3.12 or newer. [uv](https://docs.astral.sh/uv/)
is recommended:

```bash
uv sync
cp .env.example .env
```

Set `CLASH_ASSETS_DIR` in `.env` to the asset repository directory. It must
contain `troops.json`, `buildings.json`, `towers/`, `arenas/`, and the sprite
directories referenced by the metadata:

```dotenv
CLASH_ASSETS_DIR=/absolute/path/to/clash_assets
```

The dataset-builder scripts load `.env` from the repository root, so this works even when a script is launched from another directory.

## Generate a synthetic dataset

Run the dataset builder from the repository root:

```bash
uv run python dataset_builder/dataset_builder.py
```

It writes YOLO pose images, labels, and `data.yaml` to `dataset/`. By default,
it generates 10,000 training images, 1,000 validation images, no test images,
and up to 200 negative training examples from a local `negatives/` directory.
The default test split is empty. Adjust the constants in
`dataset_builder/dataset_builder.py` for smaller development runs.

**Warning:** each run recreates the configured output directory (`dataset/`
by default), replacing any existing contents.

To inspect the arena coordinate system:

```bash
uv run python dataset_builder/show_coordinates.py
```

## Dataset-builder layout

- `dataset_builder/` — synthetic scene generation, rendering, augmentation and YOLO pose-label generation.
