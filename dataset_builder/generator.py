from pathlib import Path
from random import randint, uniform, choice, choices, random
import numpy as np
from PIL import Image, ImageDraw

from camera import Camera
from asset_manager import AssetManager
from scene import AssetType, Scene, Sprite, Team, TroopAsset, Keypoint, TroopRenderState
from constants import (
    TOWER_SPECS,
    FORBIDDEN_AREAS,
    BUILDING_FORBIDDEN_TILES,
    SWARM_CONFIGS,
    SWARM_WEIGHTS,
)
from renderer import Renderer
from augmentations import RenderAugmentations
from config import get_asset_dir
import math


class Generator:
    def __init__(
        self,
        resolution: tuple[int, int],
        augmentations: RenderAugmentations | None = None,
        class_name_to_id: dict[str, int] | None = None,
        asset_dir: str | Path | None = None,
    ):
        self.resolution = resolution
        self.camera = Camera(
            width=resolution[0],
            height=resolution[1],
            arena_left=11,
            arena_bottom=844,
            tile_width=28.7,
            tile_height=23.15,
        )
        self.asset_manager = AssetManager(
            get_asset_dir(asset_dir),
            class_name_to_id=class_name_to_id,
        )
        self.augmentations = augmentations or RenderAugmentations()
        self.renderer = Renderer(self.camera, self.asset_manager, self.augmentations)

        self._ground_collision_circles: list[tuple[float, float, float]] = []
        self._air_collision_circles: list[tuple[float, float, float]] = []

        self._troop_spawn_counts: dict[str, int] = {}
        self._placement_map = np.ones((18, 32), dtype=np.float32)
        self._occupied_building_tiles: set[tuple[int, int]] = set()

    def _register_troop_collision(
        self,
        troop_asset: TroopAsset,
        x: float,
        y: float,
    ) -> None:
        collision_circle = (
            self._air_collision_circles
            if troop_asset.air
            else self._ground_collision_circles
        )

        collision_circle.append(
            (
                x,
                y,
                troop_asset.collision_radius,
            )
        )

    def generate_swarm(self) -> list[Sprite]:
        """Add one dense group of the same troop to the scene."""

        troop_name = choices(
            list(SWARM_WEIGHTS.keys()),
            weights=list(SWARM_WEIGHTS.values()),
            k=1,
        )[0]

        min_count, max_count = SWARM_CONFIGS[troop_name]
        count = randint(min_count, max_count)

        troop = self.asset_manager.get_troop(troop_name)

        # Pick a random centre.
        center_x = uniform(2.0, 16.0)
        center_y = uniform(3.0, 29.0)

        # How far the swarm is allowed to spread.
        radius = uniform(0.8, 3.0)

        troops = []
        team = choice((Team.RED, Team.BLUE))

        for _ in range(count):
            placed = False

            # More attempts are useful for dense swarms.
            for _ in range(100):

                angle = uniform(0, 2 * math.pi)
                distance = uniform(0, radius)

                x = center_x + math.cos(angle) * distance
                y = center_y + math.sin(angle) * distance

                if not self._is_valid_troop_position(
                    x,
                    y,
                    troop,
                ):
                    continue

                sprite = Sprite(
                    asset=troop,
                    image_index=troop.random_image_index(team),
                    world_x=x,
                    world_y=y,
                    team=team,
                    layer=2 if troop.air else 1,
                    render_state=self._sample_troop_render_state(),
                )

                self._register_troop_collision(
                    troop,
                    x,
                    y,
                )

                troops.append(sprite)
                placed = True
                break

            if not placed:
                continue

        return troops

    def _sample_troop_render_state(self) -> TroopRenderState:
        return TroopRenderState(
            damaged=random() < self.augmentations.damaged_probability,
            show_hp_bar=random() < self.augmentations.hp_bar_probability,
            show_level=random() < self.augmentations.level_indicator_probability,
        )

    def _building_tiles(
        self,
        origin_x: int,
        origin_y: int,
        tile_size: int,
    ) -> list[tuple[int, int]]:
        return [
            (x, y)
            for x in range(origin_x, origin_x + tile_size)
            for y in range(origin_y, origin_y + tile_size)
        ]

    def _can_place_building(
        self,
        origin_x: int,
        origin_y: int,
        tile_size: int,
    ) -> bool:
        if origin_x < 0 or origin_y < 0:
            return False

        if origin_x + tile_size > 18:
            return False

        if origin_y + tile_size > 32:
            return False

        for tile in self._building_tiles(
            origin_x,
            origin_y,
            tile_size,
        ):
            if (
                tile in self._occupied_building_tiles
                or tile in BUILDING_FORBIDDEN_TILES
            ):
                return False

        return True

    def _reserve_building(
        self,
        origin_x: int,
        origin_y: int,
        tile_size: int,
    ) -> None:
        self._occupied_building_tiles.update(
            self._building_tiles(
                origin_x,
                origin_y,
                tile_size,
            )
        )

    @staticmethod
    def _in_forbidden_corner(x: float, y: float) -> bool:
        for min_x, max_x, min_y, max_y in FORBIDDEN_AREAS:
            if min_x <= x <= max_x and min_y <= y < max_y:
                return True
        return False

    def _is_valid_troop_position(
        self,
        x: float,
        y: float,
        troop_asset: TroopAsset,
    ) -> bool:

        if not (0.5 <= x <= 17.5 and 0.5 <= y <= 32.5):
            return False

        if self._in_forbidden_corner(x, y):
            return False

        collision_circles = (
            self._air_collision_circles
            if troop_asset.air
            else self._ground_collision_circles
        )

        for tx, ty, tr in collision_circles:
            min_distance = troop_asset.collision_radius + tr
            if (x - tx) ** 2 + (y - ty) ** 2 < min_distance**2:
                return False

        return True

    def _sample_from_map(
        self,
        troop_asset: TroopAsset,
        center: tuple[float, float] | None = None,
        spread: float = 1.0,
    ) -> tuple[float, float] | None:

        if self._placement_map.sum() == 0:
            self._placement_map = np.ones((18, 32), dtype=np.float32)

        if center is None:
            valid_cells = np.argwhere(self._placement_map > 0)
        else:
            cx, cy = center

            rows, cols = np.indices(self._placement_map.shape)

            valid_cells = np.argwhere(
                (self._placement_map > 0)
                & ((rows + 0.5 - cx) ** 2 + (cols + 0.5 - cy) ** 2 <= spread**2)
            )

        if len(valid_cells) == 0:
            return None

        candidates = np.random.permutation(valid_cells)

        for gx, gy in candidates:
            x = (
                gx
                + 0.5
                + np.clip(
                    np.random.randn() * 0.2,
                    -0.4,
                    0.4,
                )
            )
            y = (
                gy
                + 0.5
                + np.clip(
                    np.random.randn() * 0.2,
                    -0.4,
                    0.4,
                )
            )

            if self._is_valid_troop_position(x, y, troop_asset):
                self._placement_map[gx, gy] = 0.0
                return x, y

        return None

    def tower_sprites(self) -> tuple[list[Sprite], set[str]]:
        sprites: list[Sprite] = []
        destroyed_towers: set[str] = set()

        for spec in TOWER_SPECS:
            is_destroyed = spec.is_princess and np.random.random() < 0.25
            if is_destroyed:
                destroyed_towers.add(spec.tower_id)
                asset = self.asset_manager.get_asset(AssetType.TOWER, "princess_ruins")
                sprites.append(
                    Sprite(
                        asset=asset,
                        image_index=asset.random_image_index(),
                        world_x=spec.world_x,
                        world_y=spec.world_y,
                        layer=-1,
                    )
                )
                continue

            asset = self.asset_manager.get_tower(spec.name)
            sprites.append(
                Sprite(
                    asset=asset,
                    image_index=asset.random_image_index(),
                    world_x=spec.world_x,
                    world_y=spec.world_y,
                    screen_x=self.camera.world_to_screen(spec.screen_x, spec.screen_y)[
                        0
                    ],
                    screen_y=self.camera.world_to_screen(spec.screen_x, spec.screen_y)[
                        1
                    ],
                    layer=spec.layer,
                )
            )

            self._ground_collision_circles.append(
                (
                    spec.world_x,
                    spec.world_y,
                    spec.collision_radius,
                )
            )

            if spec.hp_bar and spec.hp_x is not None and spec.hp_y is not None:
                hp_asset = self.asset_manager.get_asset(AssetType.UI, spec.hp_bar)
                hp_sx, hp_sy = self.camera.world_to_screen(spec.hp_x, spec.hp_y)
                sprites.append(
                    Sprite(
                        asset=hp_asset,
                        image_index=hp_asset.random_image_index(),
                        world_x=spec.world_x,
                        world_y=spec.world_y,
                        screen_x=hp_sx,
                        screen_y=hp_sy,
                        layer=spec.hp_layer,
                    )
                )

        return sprites, destroyed_towers

    def hud(self) -> list[Sprite]:
        deck = self.asset_manager.get_asset(AssetType.HUD, "deck")
        return [
            Sprite(
                asset=deck,
                image_index=deck.random_image_index(),
                world_x=-1,
                world_y=-1,
                screen_x=self.resolution[0] // 2,
                screen_y=self.resolution[1] - (184 // 2),
                layer=4,
            )
        ]

    def _scatter_items(
        self, asset_type: AssetType, count_range: tuple[int, int], layer: int
    ) -> list[Sprite]:
        sprites = []
        for _ in range(randint(*count_range)):
            asset = (
                self.asset_manager.random_background_item()
                if asset_type is AssetType.BACKGROUND_ITEM
                else self.asset_manager.random_foreground_item()
            )
            sprites.append(
                Sprite(
                    asset=asset,
                    image_index=asset.random_image_index(),
                    world_x=uniform(0, 18),
                    world_y=uniform(0, 33),
                    layer=layer,
                    render_scale=uniform(0.8, 1.2),
                )
            )
        return sprites

    def background_items(self) -> list[Sprite]:
        return self._scatter_items(AssetType.BACKGROUND_ITEM, (0, 30), 0)

    def foreground_items(self) -> list[Sprite]:
        return self._scatter_items(AssetType.FOREGROUND_ITEM, (0, 30), 3)

    def random_troop_sprite(self, team: Team | None = None) -> Sprite | None:
        if team is None:
            team = choice((Team.RED, Team.BLUE))

        troop = self.asset_manager.random_troop()

        for _ in range(50):
            pos = self._sample_from_map(troop)
            if pos:
                x, y = pos
                if troop.air:
                    self._air_collision_circles.append(
                        (
                            x,
                            y,
                            troop.collision_radius,
                        )
                    )
                else:
                    self._ground_collision_circles.append(
                        (
                            x,
                            y,
                            troop.collision_radius,
                        )
                    )

                return Sprite(
                    asset=troop,
                    image_index=troop.random_image_index(team),
                    world_x=x,
                    world_y=y,
                    layer=2 if troop.air else 1,
                    team=team,
                    render_state=self._sample_troop_render_state(),
                )
        return None

    def random_building_sprite(self) -> Sprite | None:
        building = self.asset_manager.random_building()

        for _ in range(50):
            origin_x = randint(0, 18 - building.tile_size)
            origin_y = randint(0, 32 - building.tile_size)

            if not self._can_place_building(
                origin_x,
                origin_y,
                building.tile_size,
            ):
                continue

            self._reserve_building(
                origin_x,
                origin_y,
                building.tile_size,
            )

            world_x = origin_x + building.tile_size / 2
            world_y = origin_y + building.tile_size / 2

            self._ground_collision_circles.append(
                (
                    world_x,
                    world_y,
                    building.collision_radius,
                )
            )

            return Sprite(
                asset=building,
                image_index=building.random_image_index(),
                world_x=world_x,
                world_y=world_y,
                layer=1,
            )

        return None

    def random_scene(self) -> Scene:
        # Reset all per-scene placement state
        self._ground_collision_circles.clear()
        self._air_collision_circles.clear()
        self._occupied_building_tiles.clear()

        arena = self.asset_manager.random_arena()

        tower_sprites, destroyed_ids = self.tower_sprites()

        self._placement_map = np.ones(
            (18, 32),
            dtype=np.float32,
        )

        buildings: list[Sprite] = []
        troops: list[Sprite] = []

        SCENE_TYPES = {
            "clean": 0.5,
            "moderate": 0.30,
            "crowded": 0.2,
            "swarm": 0.05,
        }


        scene_type = choices(
            list(SCENE_TYPES.keys()),
            weights=list(SCENE_TYPES.values()),
            k=1,
        )[0]

        if scene_type == "clean":
            num_buildings = randint(0, 1)

        elif scene_type == "moderate":
            num_buildings = randint(0, 2)

        elif scene_type == "crowded":
            num_buildings = randint(1, 4)

        elif scene_type == "swarm":
            num_buildings = randint(1, 5)

        for _ in range(num_buildings):
            sprite = self.random_building_sprite()

            if sprite is not None:
                buildings.append(sprite)

        if scene_type == "clean":
            num_groups = randint(1, 3)

            for _ in range(num_groups):
                sprite = self.random_troop_sprite()

                if sprite is not None:
                    troops.append(sprite)

        elif scene_type == "moderate":
            num_groups = randint(3, 7)

            for _ in range(num_groups):

                if random() < 0.05:
                    troops += self.generate_swarm()
                else:
                    sprite = self.random_troop_sprite()

                    if sprite is not None:
                        troops.append(sprite)

        elif scene_type == "crowded":
            num_groups = randint(7, 30)

            for _ in range(num_groups):

                if random() < 0.05:
                    troops += self.generate_swarm()
                else:
                    sprite = self.random_troop_sprite()

                    if sprite is not None:
                        troops.append(sprite)

        elif scene_type == "swarm":
            num_groups = randint(4, 8)

            for _ in range(num_groups):

                if random() < 0.3:
                    troops += self.generate_swarm()
                else:
                    sprite = self.random_troop_sprite()

                    if sprite is not None:
                        troops.append(sprite)

        return Scene(
            arena=arena,
            sprites=(
                tower_sprites
                + self.hud()
                + buildings
                + troops
                + self.background_items()
                + self.foreground_items()
            ),
        )

    def draw_collision_ellipse(self, frame, world_x, world_y, radius, colour):
        draw = ImageDraw.Draw(frame)

        # Centre of the collision circle in screen space.
        cx, cy = self.camera.world_to_screen(world_x, world_y)

        # Convert the radius along the world X and Y axes.
        x1, _ = self.camera.world_to_screen(world_x - radius, world_y)
        x2, _ = self.camera.world_to_screen(world_x + radius, world_y)

        _, y1 = self.camera.world_to_screen(world_x, world_y - radius)
        _, y2 = self.camera.world_to_screen(world_x, world_y + radius)

        rx = abs(x2 - x1) / 2
        ry = abs(y2 - y1) / 2

        draw.ellipse(
            [
                cx - rx,
                cy - ry,
                cx + rx,
                cy + ry,
            ],
            outline=colour,
            width=2,
        )

    def _draw_collision_debug(self, frame):
        print(
            f"Drawing {len(self._air_collision_circles)} air and {len(self._ground_collision_circles)} ground collision circles"
        )
        for world_x, world_y, radius in self._air_collision_circles:
            self.draw_collision_ellipse(frame, world_x, world_y, radius, (255, 0, 255))

        for world_x, world_y, radius in self._ground_collision_circles:
            self.draw_collision_ellipse(frame, world_x, world_y, radius, (255, 0, 0))

    def random_frame(self, debug: bool = False) -> tuple[Image.Image, list[Keypoint]]:
        scene = self.random_scene()

        frame, labels = self.renderer.render(scene, debug=debug)

        if debug:
            self._draw_collision_debug(frame)

        return frame, labels
