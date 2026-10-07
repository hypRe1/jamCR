from __future__ import annotations

from dataclasses import dataclass
from random import randint, random, uniform

import cv2
import numpy as np

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
)

from asset_manager import AssetManager
from camera import Camera
from scene import (
    AssetType,
    Keypoint,
    Scene,
    Sprite,
    SpriteImage,
    Team,
)
from effects import (
    apply_damage_tint,
    augment_colour_shift,
    augment_colour_wash,
    augment_silhouette,
    augment_translucent_effects,
    create_cr_shadow,
    partial_transparency,
)
from augmentations import RenderAugmentations

OCCLUSION_ALPHA_THRESHOLD = 32
OCCLUSION_DROP_THRESHOLD = 0.8
KEYPOINT_RADIUS = 3


@dataclass
class Attachment:
    image: Image.Image
    offset: tuple[float, float]


@dataclass
class PreparedSprite:
    """
    A sprite after all transformations needed before compositing.

    The sprite is prepared once during pass 1 and then drawn during pass 2.
    """

    sprite: Sprite
    image: Image.Image

    # Top-left position of image in the frame
    image_x: int
    image_y: int

    # Sprite anchor position in screen coordinates
    screen_x: float
    screen_y: float

    # Bbox in frame pixel coordinates
    bbox: tuple[float, float, float, float]

    # Alpha mask clipped to the frame
    mask: np.ndarray | None
    mask_box: tuple[int, int, int, int] | None

    # Attachments
    under: list[Attachment]
    over: list[Attachment]

    # Label information.
    keypoint_visible: bool
    occlusion_ratio: float


class Renderer:
    def __init__(
        self,
        camera: Camera,
        asset_manager: AssetManager,
        augmentations: RenderAugmentations,
    ):
        self.camera = camera
        self.asset_manager = asset_manager
        self.augmentations = augmentations

    def _screen_position(
        self,
        sprite: Sprite,
    ) -> tuple[float, float]:
        """
        Convert the sprite's world position to screen coordinates.

        Some sprites, such as towers/UI elements, can override the calculated
        screen position.
        """

        x, y = self.camera.world_to_screen(
            sprite.world_x,
            sprite.world_y,
        )

        if sprite.screen_x is not None:
            x = sprite.screen_x

        if sprite.screen_y is not None:
            y = sprite.screen_y

        return x, y

    def _make_silhouette(
        self,
        image: Image.Image,
        colour: tuple[int, int, int] = (128, 128, 128),
    ) -> Image.Image:
        """Replace an RGBA image with a solid-colour silhouette."""

        alpha = image.getchannel("A")

        silhouette = Image.new(
            "RGBA",
            image.size,
            (*colour, 255),
        )

        silhouette.putalpha(alpha)

        return silhouette


    def _appearance_augment(
        self,
        sprite: Sprite,
        image: Image.Image,
    ) -> Image.Image:
        """Apply appearance augmentations to troop sprites."""

        if not (sprite.is_troop or sprite.is_building):
            return image

        if random() < self.augmentations.transparent_probability:
            image = partial_transparency(image, self.augmentations.transparent_range)

        if random() < self.augmentations.silhouette_probability:
            return augment_silhouette(image)

        if random() < self.augmentations.colour_shift_probability:
            return augment_colour_shift(image)

        return image

    def _get_sprite_image(
        self,
        sprite: Sprite,
    ) -> Image.Image:
        """Select the sprite frame and optionally flip it."""

        sprite_image = sprite.asset.get_image(sprite.image_index)

        if sprite.is_troop and random() < self.augmentations.flip_probability:
            sprite_image = sprite_image.flipped()

        return sprite_image.image

    def _transform_image(
        self,
        sprite: Sprite,
        image: Image.Image,
    ) -> tuple[Image.Image, float]:
        """Apply sprite scale and small stretch variation."""

        base_scale = (
            sprite.render_scale
            if sprite.render_scale is not None
            else sprite.asset.render_scale
        )

        scale = base_scale * uniform(*self.augmentations.scale_range)

        stretch = uniform(*self.augmentations.stretch_range)

        stretch_x = stretch * uniform(0.98, 1.02)
        stretch_y = stretch * uniform(0.98, 1.02)

        width = max(
            1,
            round(image.width * scale * stretch_x),
        )

        height = max(
            1,
            round(image.height * scale * stretch_y),
        )

        if (width, height) != image.size:
            image = image.resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )

        return image, scale

    @staticmethod
    def _colour_augment(
        image: Image.Image,
        brightness: float,
        saturation: float,
        contrast: float,
    ) -> Image.Image:
        """Apply brightness, saturation and contrast to an RGBA image."""

        rgba = image.convert("RGBA")
        rgb = rgba.convert("RGB")

        if brightness != 1.0:
            rgb = ImageEnhance.Brightness(rgb).enhance(brightness)

        if saturation != 1.0:
            rgb = ImageEnhance.Color(rgb).enhance(saturation)

        if contrast != 1.0:
            rgb = ImageEnhance.Contrast(rgb).enhance(contrast)

        return Image.merge(
            "RGBA",
            (
                rgb.getchannel("R"),
                rgb.getchannel("G"),
                rgb.getchannel("B"),
                rgba.getchannel("A"),
            ),
        )

    def _maybe_colour_augment(
        self,
        sprite: Sprite,
        image: Image.Image,
    ) -> Image.Image:
        """Randomly modify the appearance of an asset."""

        if sprite.asset.asset_type not in {
            AssetType.TROOP,
            AssetType.TOWER,
            AssetType.BACKGROUND_ITEM,
            AssetType.FOREGROUND_ITEM,
        }:
            return image

        probability = getattr(
            self.augmentations,
            "asset_colour_probability",
            0.35,
        )

        if random() >= probability:
            return image

        brightness_range = getattr(
            self.augmentations,
            "asset_brightness",
            self.augmentations.brightness,
        )

        saturation_range = getattr(
            self.augmentations,
            "asset_saturation",
            self.augmentations.saturation,
        )

        contrast_range = getattr(
            self.augmentations,
            "asset_contrast",
            self.augmentations.contrast,
        )

        return self._colour_augment(
            image,
            uniform(*brightness_range),
            uniform(*saturation_range),
            uniform(*contrast_range),
        )

    def _create_attachments(
        self,
        sprite: Sprite,
        image: Image.Image,
        scale: float,
    ) -> tuple[list[Attachment], list[Attachment]]:
        """
        Create things attached to a troop:

        - shadow underneath
        - HP bar
        - level indicator
        """

        if not sprite.is_troop:
            return [], []

        under = []
        over = []

        if random() < self.augmentations.shadow_probability:
            shadow, _ = create_cr_shadow(
                image,
                is_air=sprite.troop_asset.air,
            )

            under.append(
                Attachment(
                    image=shadow,
                    offset=(0, 0),
                )
            )

        state = sprite.render_state

        if state is None:
            return under, over

        if not state.show_hp_bar and not state.show_level:
            return under, over

        side = "player" if sprite.team is Team.BLUE else "enemy"

        if state.show_hp_bar and state.show_level:
            show_level = random() > 0.5
        else:
            show_level = state.show_level

        if show_level:
            asset_name = f"level_{side}"
        else:
            asset_name = f"hp_{side}_{sprite.troop_asset.hp_bar}"

        ui_asset = self.asset_manager.get_asset(
            AssetType.UI,
            asset_name,
        )

        ui_image = ui_asset.images[ui_asset.random_image_index()].image

        offset_y = -sprite.troop_asset.hp_offset * scale * uniform(0.8, 1.3)

        over.append(
            Attachment(
                image=ui_image,
                offset=(0, offset_y),
            )
        )

        return under, over

    @staticmethod
    def _draw_attachment(
        frame: Image.Image,
        attachment: Attachment,
        x: float,
        y: float,
    ) -> None:
        """Draw an attachment centred around x/y."""

        px = int(x - attachment.image.width / 2 + attachment.offset[0])

        py = int(y - attachment.image.height / 2 + attachment.offset[1])

        frame.alpha_composite(
            attachment.image,
            (px, py),
        )

    @staticmethod
    def _get_alpha_mask(
        image: Image.Image,
    ) -> np.ndarray:
        """Convert the sprite alpha channel into a binary mask."""

        alpha = np.asarray(
            image.getchannel("A"),
            dtype=np.uint8,
        )

        return alpha >= OCCLUSION_ALPHA_THRESHOLD

    def _get_bbox_from_mask(
        self,
        mask: np.ndarray,
        screen_x: float,
        screen_y: float,
    ) -> tuple[float, float, float, float] | None:
        """
        Calculate the visible sprite bbox from an already-created mask.
        """

        ys, xs = np.where(mask)

        if len(xs) == 0:
            return None

        left = xs.min()
        right = xs.max() + 1
        top = ys.min()
        bottom = ys.max() + 1

        image_height, image_width = mask.shape

        image_x = screen_x - image_width / 2

        image_y = screen_y - image_height / 2

        x1 = max(
            0.0,
            image_x + left,
        )

        y1 = max(
            0.0,
            image_y + top,
        )

        x2 = min(
            float(self.camera.width),
            image_x + right,
        )

        y2 = min(
            float(self.camera.height),
            image_y + bottom,
        )

        if x1 >= x2 or y1 >= y2:
            return None

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        )

    @staticmethod
    def _clip_mask(
        mask: np.ndarray,
        image_x: int,
        image_y: int,
        frame_width: int,
        frame_height: int,
    ) -> (
        tuple[
            np.ndarray,
            int,
            int,
            int,
            int,
        ]
        | None
    ):
        """Clip an image mask to the visible frame."""

        x1 = max(
            0,
            image_x,
        )

        y1 = max(
            0,
            image_y,
        )

        x2 = min(
            frame_width,
            image_x + mask.shape[1],
        )

        y2 = min(
            frame_height,
            image_y + mask.shape[0],
        )

        if x1 >= x2 or y1 >= y2:
            return None

        mask_x1 = x1 - image_x
        mask_y1 = y1 - image_y

        mask_x2 = mask_x1 + (x2 - x1)

        mask_y2 = mask_y1 + (y2 - y1)

        return (
            mask[
                mask_y1:mask_y2,
                mask_x1:mask_x2,
            ],
            x1,
            y1,
            x2,
            y2,
        )

    @staticmethod
    def _occlusion_ratio(
        occupied: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Return the fraction of the sprite covered by earlier sprites."""

        total_pixels = np.count_nonzero(mask)

        if total_pixels == 0:
            return 0.0

        covered_pixels = np.count_nonzero(occupied & mask)

        return covered_pixels / total_pixels

    @staticmethod
    def _keypoint_visible(
        occupied: np.ndarray,
        x: int,
        y: int,
    ) -> bool:
        """
        Check whether the pivot is sufficiently unobstructed.

        A small region around the pivot is used rather than a single pixel.
        """

        height, width = occupied.shape

        if not (0 <= x < width and 0 <= y < height):
            return True

        radius = KEYPOINT_RADIUS

        x1 = max(
            0,
            x - radius,
        )

        y1 = max(
            0,
            y - radius,
        )

        x2 = min(
            width,
            x + radius + 1,
        )

        y2 = min(
            height,
            y + radius + 1,
        )

        region = occupied[
            y1:y2,
            x1:x2,
        ]

        if region.size == 0:
            return True

        occupied_fraction = np.count_nonzero(region) / region.size

        return occupied_fraction <= 0.5

    def _prepare_sprite(
        self,
        sprite: Sprite,
    ) -> PreparedSprite | None:
        """
        Prepare a sprite for rendering.

        This handles:

        1. Position
        2. Image selection
        3. Scaling
        4. Colour augmentation
        5. Damage
        6. Attachments
        7. Alpha mask
        8. Bounding box
        """

        screen_x, screen_y = self._screen_position(sprite)

        image = self._get_sprite_image(sprite)

        image, scale = self._transform_image(
            sprite,
            image,
        )

        image = self._maybe_colour_augment(
            sprite,
            image,
        )

        image = self._appearance_augment(
            sprite,
            image,
        )

        if (sprite.is_troop or sprite.is_building) and sprite.render_state and sprite.render_state.damaged:
            image, _ = apply_damage_tint(
                image,
                sprite.team,
                1,
            )

        under, over = self._create_attachments(
            sprite,
            image,
            scale,
        )

        image_x = int(screen_x - image.width / 2)
        image_y = int(screen_y - image.height / 2)

        mask = self._get_alpha_mask(image)

        bbox = self._get_bbox_from_mask(
            mask,
            screen_x,
            screen_y,
        )

        if bbox is None:
            return None

        clipped = self._clip_mask(
            mask,
            image_x,
            image_y,
            self.camera.width,
            self.camera.height,
        )

        if clipped is None:
            clipped_mask = None
            mask_box = None
        else:
            (
                clipped_mask,
                x1,
                y1,
                x2,
                y2,
            ) = clipped

            mask_box = (
                x1,
                y1,
                x2,
                y2,
            )

        return PreparedSprite(
            sprite=sprite,
            image=image,
            image_x=image_x,
            image_y=image_y,
            screen_x=screen_x,
            screen_y=screen_y,
            bbox=bbox,
            mask=clipped_mask,
            mask_box=mask_box,
            under=under,
            over=over,
            keypoint_visible=True,
            occlusion_ratio=0.0,
        )

    def _process_occlusion(
        self,
        item: PreparedSprite,
        occupied: np.ndarray,
    ) -> bool:
        """
        Calculate occlusion and keypoint visibility.

        Returns False if the sprite should not receive a label/render.
        """

        sprite = item.sprite

        # Only troops have YOLO labels.
        if not (
            (sprite.is_troop or sprite.is_building)
            and sprite.asset.class_id is not None
        ):
            return True

        if item.mask is None or item.mask_box is None:
            return True

        x1, y1, x2, y2 = item.mask_box

        occupied_region = occupied[
            y1:y2,
            x1:x2,
        ]

        item.occlusion_ratio = self._occlusion_ratio(
            occupied_region,
            item.mask,
        )

        if item.occlusion_ratio >= OCCLUSION_DROP_THRESHOLD:
            return False

        item.keypoint_visible = self._keypoint_visible(
            occupied,
            int(item.screen_x),
            int(item.screen_y),
        )

        return True

    def _occupy(
        self,
        item: PreparedSprite,
        occupied: np.ndarray,
    ) -> None:
        """Add a sprite's visible alpha mask to the occupancy map."""

        if item.mask is None or item.mask_box is None:
            return

        x1, y1, x2, y2 = item.mask_box

        occupied[
            y1:y2,
            x1:x2,
        ] |= item.mask

    def _make_label(
        self,
        item: PreparedSprite,
    ) -> Keypoint | None:
        """Create the YOLO pose label for a prepared troop."""

        sprite = item.sprite

        if sprite.asset.class_id is None:
            return None

        bbox_x, bbox_y, bbox_w, bbox_h = item.bbox

        return Keypoint(
            class_id=sprite.asset.class_id,
            bbox_x=(bbox_x / self.camera.width),
            bbox_y=(bbox_y / self.camera.height),
            bbox_w=(bbox_w / self.camera.width),
            bbox_h=(bbox_h / self.camera.height),
            kpt_x=(item.screen_x / self.camera.width),
            kpt_y=(item.screen_y / self.camera.height),
            kpt_visible=item.keypoint_visible,
            occlusion_ratio=item.occlusion_ratio,
        )

    def _draw_prepared_sprite(
        self,
        frame: Image.Image,
        item: PreparedSprite,
        ui_elements: list,
        silhouettes: bool = False,
    ) -> None:
        """
        Draw one prepared sprite and collect its over-attachments.
        """

        sprite = item.sprite

        for attachment in item.under:
            self._draw_attachment(
                frame,
                attachment,
                item.screen_x,
                item.screen_y,
            )

        image = item.image
        if silhouettes and (sprite.is_troop or sprite.is_building):
            image = self._make_silhouette(image)

        if sprite.is_troop or sprite.is_building:
            if random() < self.augmentations.translucent_effect_probability:
                image = augment_translucent_effects(
                    image,
                    self.augmentations.translucent_effect_count,
                    self.augmentations.translucent_effect_radius,
                    self.augmentations.translucent_effect_alpha,
                )

            if random() < self.augmentations.colour_wash_probability:
                image = augment_colour_wash(
                    image,
                    self.augmentations.colour_wash_alpha,
                )

        if (sprite.is_troop or sprite.is_building) and random() < self.augmentations.blur_probability:
            alpha = image.getchannel("A").filter(
                ImageFilter.GaussianBlur(uniform(*self.augmentations.alpha_blur))
            )

            image = image.copy()
            image.putalpha(alpha)

        frame.alpha_composite(
            image,
            (
                item.image_x,
                item.image_y,
            ),
        )

        for attachment in item.over:
            ui_elements.append(
                (
                    attachment,
                    item.screen_x,
                    item.screen_y,
                )
            )

    def _augment_frame(
        self,
        frame: Image.Image,
    ) -> Image.Image:
        """
        Apply camera/image-level augmentation.

        NOTE:
        This is intentionally kept separate from sprite rendering.
        """

        rgb = np.asarray(
            frame.convert("RGB"),
            dtype=np.uint8,
        )

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        quality = randint(*self.augmentations.jpeg_quality)

        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                quality,
            ],
        )

        if ok:
            decoded = cv2.imdecode(
                encoded,
                cv2.IMREAD_COLOR,
            )

            if decoded is not None:
                rgb = cv2.cvtColor(
                    decoded,
                    cv2.COLOR_BGR2RGB,
                )

                frame = Image.fromarray(
                    rgb,
                    "RGB",
                )

        if random() > 0.3:
            frame = ImageEnhance.Brightness(frame).enhance(
                uniform(*self.augmentations.brightness)
            )

        if random() > 0.3:
            frame = ImageEnhance.Color(frame).enhance(
                uniform(*self.augmentations.saturation)
            )

        if random() > 0.5:
            frame = ImageEnhance.Contrast(frame).enhance(
                uniform(*self.augmentations.contrast)
            )

        return frame

    def render(
        self,
        scene: Scene,
        debug: bool = False,
        silhouettes: bool = False,
    ) -> tuple[Image.Image, list[Keypoint]]:
        frame = scene.arena.copy()

        labels: list[Keypoint] = []

        occupied = np.zeros(
            (
                frame.height,
                frame.width,
            ),
            dtype=np.bool_,
        )

        sprites = sorted(
            scene.sprites,
            key=lambda sprite: (
                sprite.layer,
                -sprite.world_y,
            ),
        )

        prepared: list[PreparedSprite] = []

        for sprite in reversed(sprites):
            item = self._prepare_sprite(sprite)

            if item is None:
                continue

            if not self._process_occlusion(
                item,
                occupied,
            ):
                continue

            prepared.append(item)

            self._occupy(
                item,
                occupied,
            )

        ui_elements = []

        for item in reversed(prepared):
            self._draw_prepared_sprite(
                frame,
                item,
                ui_elements,
                silhouettes=silhouettes
            )

            label = self._make_label(item)

            if label is not None:
                labels.append(label)

        for attachment, x, y in ui_elements:
            self._draw_attachment(
                frame,
                attachment,
                x,
                y,
            )

        frame = self._augment_frame(frame)

        return frame, labels
