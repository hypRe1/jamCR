from dataclasses import dataclass

@dataclass
class RenderAugmentations:
    # geometry
    scale_range: tuple[float, float] = (0.90, 1.10)
    stretch_range: tuple[float, float] = (0.90, 1.10)
    flip_probability: float = 0.5

    # colour
    brightness: tuple[float, float] = (0.80, 1.20)
    contrast: tuple[float, float] = (0.80, 1.20)
    saturation: tuple[float, float] = (0.75, 1.25)
    asset_colour_probability: float = 0.50

    # camera
    jpeg_quality: tuple[int, int] = (75, 95)
    blur_probability: float = 0.08
    alpha_blur: tuple[float, float] = (0.4, 0.9)

    # troop states
    damaged_probability: float = 0.15
    hp_bar_probability: float = 0.40
    level_indicator_probability: float = 0.25

    # translucent effects
    translucent_effect_probability: float = 0.08
    translucent_effect_count: tuple[int, int] = (2, 8)
    translucent_effect_radius: tuple[float, float] = (6.0, 30.0)
    translucent_effect_alpha: tuple[float, float] = (0.08, 0.25)

    # colour wash
    colour_wash_probability: float = 0.08
    colour_wash_alpha: tuple[float, float] = (0.02, 0.10)

    shadow_probability: float = 0.50

    colour_shift_probability: float = 0.08
    silhouette_probability: float = 0.1

    transparent_probability: float = 0.03
    transparent_range: tuple[float, float] = (0.65, 0.90)
