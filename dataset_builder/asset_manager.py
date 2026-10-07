import json
from pathlib import Path
from random import choice

from PIL import Image

from scene import Asset, AssetType, BuildingAsset, SpriteImage, TroopAsset


def _load_png(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _load_sprite_dir(
    directory: Path,
    glob: str = "*.png",
    recursive: bool = False,
) -> list[SpriteImage]:
    method = directory.rglob if recursive else directory.glob
    return [SpriteImage.from_image(_load_png(p)) for p in sorted(method(glob))]


class AssetManager:
    def __init__(
        self,
        asset_dir: str | Path,
        troop_folder="troops",
        class_name_to_id: dict[str, int] | None = None,
    ):
        self.asset_dir = Path(asset_dir)
        self.troop_folder = troop_folder
        self._class_name_to_id = class_name_to_id
        self._troop_cache: dict[str, TroopAsset] = {}
        self._building_cache: dict[str, BuildingAsset] = {}
        self._asset_cache: dict[tuple[AssetType, str], Asset] = {}
        self._towers: dict[str, Asset] = {}
        self._arena_cache: dict[str, Image.Image] = {}
        self._troop_metadata: dict[str, dict] = {}
        self._building_metadata: dict[str, dict] = {}
        self._class_ids: dict[str, int] = {}
        self._class_names: list[str] = []

        self._load_metadata()
        self._load_towers()

    def _load_metadata(self) -> None:
        with open(self.asset_dir / "troops.json") as f:
            self._troop_metadata = json.load(f)

        with open(self.asset_dir / "buildings.json") as f:
            self._building_metadata = json.load(f)

        if self._class_name_to_id is None:
            self._class_ids = {}
            for name in self._troop_metadata:
                self._class_ids[name] = len(self._class_ids)
            for name in self._building_metadata:
                self._class_ids[name] = len(self._class_ids)
            self._class_names = [
                name
                for name, _ in sorted(
                    self._class_ids.items(),
                    key=lambda item: item[1],
                )
            ]
            return

        self._class_ids = {}
        used_ids = set(self._class_name_to_id.values())
        next_id = max(used_ids, default=-1) + 1

        for name in (*self._troop_metadata, *self._building_metadata):
            class_id = self._class_name_to_id.get(name)
            if class_id is None and name == "bat":
                class_id = self._class_name_to_id.get("bats")
            if class_id is None:
                while next_id in used_ids:
                    next_id += 1
                class_id = next_id
                used_ids.add(class_id)
                next_id += 1
            self._class_ids[name] = class_id

        names_by_id = {
            class_id: name
            for name, class_id in self._class_name_to_id.items()
        }
        for name, class_id in self._class_ids.items():
            names_by_id.setdefault(class_id, "bats" if name == "bat" else name)
        self._class_names = [
            names_by_id[class_id]
            for class_id in range(max(names_by_id) + 1)
        ]

    def _load_towers(self) -> None:
        tower_root = self.asset_dir / "towers"
        if not tower_root.exists():
            raise FileNotFoundError(tower_root)

        for tower_dir in tower_root.iterdir():
            if not tower_dir.is_dir():
                continue
            images = _load_sprite_dir(tower_dir)
            if not images:
                continue
            self._towers[tower_dir.name] = Asset(
                class_id=None,
                name=tower_dir.name,
                asset_type=AssetType.TOWER,
                render_scale=1,
                images=images,
            )

    def get_troop(self, troop_name: str) -> TroopAsset:
        if cached := self._troop_cache.get(troop_name):
            return cached

        if troop_name not in self._troop_metadata:
            raise ValueError(f"Unknown troop '{troop_name}'")

        troop_dir = self.asset_dir / self.troop_folder / troop_name
        if not troop_dir.exists():
            raise FileNotFoundError(troop_dir)

        images: list[SpriteImage] = []
        blue_idx: list[int] = []
        red_idx: list[int] = []


        for animation_dir in sorted(troop_dir.iterdir()):
            if not animation_dir.is_dir():
                continue

            # Team-specific directories are handled separately
            if animation_dir.name in ("blue", "red"):
                continue

            animation_images = _load_sprite_dir(animation_dir)

            start = len(images)
            images.extend(animation_images)

            # Neutral images can be used by both teams
            indexes = range(start, start + len(animation_images))
            blue_idx.extend(indexes)
            red_idx.extend(indexes)


        for variant, bucket in (
            ("blue", blue_idx),
            ("red", red_idx),
        ):
            variant_dir = troop_dir / variant

            if not variant_dir.is_dir():
                continue

            # Animation directories are nested inside blue/red
            variant_images = _load_sprite_dir(
                variant_dir,
                recursive=True,
            )

            start = len(images)
            images.extend(variant_images)

            bucket.extend(range(start, start + len(variant_images)))

        if not images:
            raise ValueError(f"No images found for {troop_name}")

        meta = self._troop_metadata[troop_name]

        asset = TroopAsset(
            name=troop_name,
            asset_type=AssetType.TROOP,
            class_id=self._class_ids[troop_name],
            render_scale=meta["render_scale"],
            air=meta["air"],
            hp_bar=meta["hp_bar"],
            hp_offset=meta["hp_offset"],
            images=images,
            blue_image_indexes=blue_idx,
            red_image_indexes=red_idx,
            collision_radius=meta.get("collision_radius", 0.5),
        )

        self._troop_cache[troop_name] = asset
        return asset

    def random_troop(self) -> TroopAsset:
        return self.get_troop(choice(list(self._troop_metadata)))

    def get_building(self, building_name: str) -> BuildingAsset:
        if cached := self._building_cache.get(building_name):
            return cached

        if building_name not in self._building_metadata:
            raise ValueError(f"Unknown building '{building_name}'")

        building_dir = self.asset_dir / "buildings" / building_name

        if not building_dir.exists():
            raise FileNotFoundError(building_dir)

        images = _load_sprite_dir(building_dir, recursive=True)

        if not images:
            raise ValueError(f"No images found for {building_name}")

        meta = self._building_metadata[building_name]

        asset = BuildingAsset(
            name=building_name,
            asset_type=AssetType.BUILDING,
            class_id=self._class_ids[building_name],
            render_scale=meta["scale"],
            hp_bar=meta["hp_bar"],
            hp_offset=meta["hp_offset"],
            collision_radius=meta["collision_radius"],
            tile_size=meta["tile_size"],
            images=images,
        )

        self._building_cache[building_name] = asset
        return asset

    def random_building(self) -> BuildingAsset:
        return self.get_building(
            choice(list(self._building_metadata))
        )

    def get_tower(self, tower: str) -> Asset:
        try:
            return self._towers[tower]
        except KeyError:
            raise ValueError(f"No tower called {tower}") from None

    def get_asset(self, asset_type: AssetType, name: str) -> Asset:
        key = (asset_type, name)
        if cached := self._asset_cache.get(key):
            return cached

        asset_dir = self.asset_dir / asset_type.value / name
        if not asset_dir.exists():
            raise FileNotFoundError(asset_dir)

        images = _load_sprite_dir(asset_dir, recursive=True)
        if not images:
            raise ValueError(f"No images found for {name}")

        asset = Asset(
            name=name,
            asset_type=asset_type,
            class_id=None,
            render_scale=1,
            images=images,
        )
        self._asset_cache[key] = asset
        return asset

    def _random_asset_of_type(self, asset_type: AssetType) -> Asset:
        folder = self.asset_dir / asset_type.value
        names = [p.name for p in folder.iterdir() if p.is_dir()]
        if not names:
            raise ValueError(f"No items of type {asset_type}")
        return self.get_asset(asset_type, choice(names))

    def random_background_item(self) -> Asset:
        return self._random_asset_of_type(AssetType.BACKGROUND_ITEM)

    def random_foreground_item(self) -> Asset:
        return self._random_asset_of_type(AssetType.FOREGROUND_ITEM)

    def get_arena(self, arena_name: str) -> Image.Image:
        if cached := self._arena_cache.get(arena_name):
            return cached
        path = self.asset_dir / "arenas" / f"{arena_name}.png"
        if not path.exists():
            raise FileNotFoundError(path)
        img = _load_png(path)
        self._arena_cache[arena_name] = img
        return img

    def random_arena(self) -> Image.Image:
        arenas = [p.stem for p in (self.asset_dir / "arenas").glob("*.png")]
        if not arenas:
            raise ValueError("No arenas found")
        return self.get_arena(choice(arenas))

    @property
    def class_ids(self) -> dict[str, int]:
        return dict(self._class_ids)

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)
