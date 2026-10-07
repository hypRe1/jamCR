from dataclasses import dataclass
from enum import Enum
from PIL import Image

class AssetType(Enum):
    TROOP = "troops"
    BUILDING = "buildings"
    TOWER = "towers"
    UI = "ui"
    HUD = "hud"
    BACKGROUND_ITEM = "background-items"
    FOREGROUND_ITEM = "foreground-items"

class Team(Enum):
    RED = "red"
    BLUE = "blue"

@dataclass
class TroopRenderState:
    damaged: bool = False
    show_hp_bar: bool = False
    show_level: bool = False

@dataclass
class Keypoint:
    class_id: int
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    kpt_x: float
    kpt_y: float
    kpt_visible: bool = True
    occlusion_ratio: float = 0.0

@dataclass
class SpriteImage:
    image: Image.Image
    alpha_bbox: tuple[int, int, int, int] | None = None

    @classmethod
    def from_image(cls, image: Image.Image) -> 'SpriteImage':
        if image.mode != "RGBA":
            image = image.convert("RGBA")
            
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        
        return cls(image=image, alpha_bbox=bbox)

    def flipped(self) -> "SpriteImage":
        """Flips image horizontally and correctly mirrors the alpha_bbox."""
        flipped_img = self.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        new_bbox = None
        if self.alpha_bbox is not None:
            l, t, r, b = self.alpha_bbox
            w = self.image.width
            new_bbox = (w - r, t, w - l, b)
            
        return SpriteImage(
            image=flipped_img,
            alpha_bbox=new_bbox
        )

@dataclass
class Asset:
    name: str
    asset_type: AssetType
    class_id: int | None
    render_scale: float
    images: list[SpriteImage]

    def get_image(self, index: int) -> SpriteImage:
        return self.images[index]

    def random_image_index(self) -> int:
        import random
        return random.randint(0, len(self.images) - 1)

@dataclass
class TroopAsset(Asset):
    air: bool = False
    hp_bar: str = "medium"
    hp_offset: int = -40
    collision_radius: float = 0.5
    blue_image_indexes: list[int] | None = None
    red_image_indexes: list[int] | None = None

    def random_image_index(self, team: Team | None = None) -> int:
        import random
        if team == Team.BLUE and self.blue_image_indexes:
            return random.choice(self.blue_image_indexes)
        if team == Team.RED and self.red_image_indexes:
            return random.choice(self.red_image_indexes)
        return random.randint(0, len(self.images) - 1)

@dataclass
class BuildingAsset(Asset):
    hp_bar: str = "high"
    hp_offset: int = 70
    collision_radius: float = 1
    tile_size: int = 3

@dataclass
class Sprite:
    asset: Asset
    image_index: int
    world_x: float
    world_y: float
    layer: int = 1
    screen_x: float | None = None
    screen_y: float | None = None
    team: Team | None = None
    render_scale: float | None = None
    render_state: TroopRenderState | None = None

    @property
    def is_troop(self) -> bool:
        return self.asset.asset_type is AssetType.TROOP

    @property
    def troop_asset(self) -> TroopAsset:
        if self.is_troop:
            return self.asset # type: ignore
        raise ValueError("Not a troop")

    @property
    def is_building(self) -> bool:
        return self.asset.asset_type is AssetType.BUILDING

    @property
    def building_asset(self) -> BuildingAsset:
        if self.is_building:
            return self.asset  # type: ignore
        raise ValueError("Not a building")

@dataclass
class Scene:
    arena: Image.Image
    sprites: list[Sprite]