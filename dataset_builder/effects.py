from random import random, randint, uniform, choice

import numpy as np
from PIL import Image, ImageOps, ImageChops, ImageFilter, ImageDraw

from scene import Team


def apply_damage_tint(image: Image.Image, team: Team | None, prob=0.25):
    if random() > prob:
        return image, False

    strength = randint(60, 150) / 255.0
    if team is None:
        team = choice((Team.RED, Team.BLUE))

    if team is Team.BLUE:
        tint = (randint(20, 80), randint(80, 160), randint(180, 255))
    else:
        tint = (randint(180, 255), randint(20, 80), randint(20, 80))

    arr = np.asarray(image).astype(np.float32)
    rgb, alpha = arr[..., :3], arr[..., 3:4]
    tint_arr = np.array(tint, dtype=np.float32)
    blended = rgb * (1 - strength) + tint_arr * strength
    out = np.concatenate([blended, alpha], axis=-1).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA"), True


def augment_translucent_effects(
    image: Image.Image,
    effect_count,
    effect_radius,
    effect_alpha,
) -> Image.Image:
    """
    Add semi-transparent coloured shapes over a sprite.

    The sprite's original alpha channel is preserved so that
    bbox/keypoint/occlusion geometry is unchanged.
    """

    image = image.convert("RGBA")

    original_alpha = image.getchannel("A")

    effect = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(effect)

    count = randint(*effect_count)

    for _ in range(count):
        radius = uniform(*effect_radius)

        x = uniform(0, image.width)
        y = uniform(0, image.height)

        rx = radius * uniform(0.6, 1.4)
        ry = radius * uniform(0.6, 1.4)

        alpha = randint(
            int(effect_alpha[0] * 255),
            int(effect_alpha[1] * 255),
        )

        colour = (
            randint(0, 255),
            randint(0, 255),
            randint(0, 255),
            alpha,
        )

        draw.ellipse(
            (
                x - rx,
                y - ry,
                x + rx,
                y + ry,
            ),
            fill=colour,
        )

    image = Image.alpha_composite(
        image,
        effect,
    )

    # Do not let the effect alter the sprite's shape.
    image.putalpha(original_alpha)

    return image


def augment_colour_wash(
    image: Image.Image,
    wash_alpha,
) -> Image.Image:
    """
    Apply a semi-transparent random colour wash over a sprite.
    """

    image = image.convert("RGBA")

    original_alpha = image.getchannel("A")

    colour = (
        randint(0, 255),
        randint(0, 255),
        randint(0, 255),
        randint(
            int(wash_alpha[0] * 255),
            int(wash_alpha[1] * 255),
        ),
    )

    effect = Image.new(
        "RGBA",
        image.size,
        colour,
    )

    image = Image.alpha_composite(
        image,
        effect,
    )

    image.putalpha(original_alpha)

    return image


def create_cr_shadow(
    sprite_img: Image.Image,
    is_air: bool = False,
    pad_ratio: float = 0.8,
) -> tuple[Image.Image, int]:
    pad = int(max(sprite_img.width, sprite_img.height) * pad_ratio)
    width = sprite_img.width + pad * 2
    height = sprite_img.height + pad * 2
    cx, cy = width / 2, height / 2

    alpha = Image.new("L", (width, height), 0)
    alpha.paste(sprite_img.split()[3], (pad, pad))

    opacity = randint(100, 190)
    shadow_alpha = ImageChops.darker(alpha, Image.new("L", (width, height), opacity))

    tint_color = (randint(30, 50), randint(40, 60), randint(70, 100), 255)
    silhouette = Image.new("RGBA", (width, height), color=tint_color)
    silhouette.putalpha(shadow_alpha)
    flipped = ImageOps.flip(silhouette)

    scale_x = uniform(0.8, 1.0)
    scale_y = uniform(0.55, 0.85)
    skew_x = uniform(-0.55, -0.35)
    drop_y = height * (uniform(0.08, 0.12) if is_air else uniform(0.0, 0.02))

    a = 1.0 / scale_x
    b = -skew_x / (scale_x * scale_y)
    c = cx - cx / scale_x + skew_x * (cy + drop_y) / (scale_x * scale_y)
    d = 0.0
    e = 1.0 / scale_y
    f = cy - (cy + drop_y) / scale_y

    projected = flipped.transform(
        (width, height),
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.BILINEAR,
    )
    projected = projected.filter(ImageFilter.GaussianBlur(radius=uniform(0.8, 1.5)))
    return projected, pad


def augment_colour_shift(
    image: Image.Image,
    hue_shift: tuple[float, float] = (-0.08, 0.08),
    saturation_scale: tuple[float, float] = (0.7, 1.3),
    brightness_scale: tuple[float, float] = (0.75, 1.25),
) -> Image.Image:
    """
    Shift the colours of a sprite while preserving its alpha/geometry.

    The hue is rotated, while saturation and brightness are independently
    varied. This changes the exact appearance of the sprite without changing
    its silhouette.
    """

    image = image.convert("RGBA")

    rgba = np.asarray(image, dtype=np.uint8)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    # Convert RGB -> HSV.
    hsv = np.asarray(
        Image.fromarray(rgb, mode="RGB").convert("HSV"),
        dtype=np.uint8,
    ).astype(np.float32)

    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    brightness = hsv[..., 2]

    # Hue in PIL HSV is represented in [0, 255].
    hue += uniform(*hue_shift) * 255.0
    hue %= 255.0

    saturation *= uniform(*saturation_scale)
    brightness *= uniform(*brightness_scale)

    hsv[..., 0] = hue
    hsv[..., 1] = np.clip(saturation, 0, 255)
    hsv[..., 2] = np.clip(brightness, 0, 255)

    shifted = Image.fromarray(
        hsv.astype(np.uint8),
        mode="HSV",
    ).convert("RGB")

    result = np.asarray(shifted, dtype=np.uint8)

    return Image.fromarray(
        np.dstack((result, alpha)),
        mode="RGBA",
    )


def augment_silhouette(
    image: Image.Image,
) -> Image.Image:
    """
    Replace the visible sprite with a single random solid colour.

    The original alpha channel is preserved exactly, meaning the sprite's
    silhouette and geometry remain unchanged.
    """

    image = image.convert("RGBA")

    alpha = image.getchannel("A")

    colour = (
        randint(0, 255),
        randint(0, 255),
        randint(0, 255),
        255,
    )

    silhouette = Image.new(
        "RGBA",
        image.size,
        colour,
    )

    silhouette.putalpha(alpha)

    return silhouette


def partial_transparency(image: Image.Image, alpha_range: tuple[float, float]) -> Image.Image:
    """
    Apply partial transparency to an RGBA sprite.

    Returns the modified image.
    """
    alpha = uniform(
        *alpha_range
    )

    image = image.copy()

    if image.mode != "RGBA":
        image = image.convert("RGBA")

    r, g, b, a = image.split()

    # Reduce existing alpha rather than replacing it.
    a = a.point(
        lambda value: int(value * alpha)
    )

    image.putalpha(a)

    return image
