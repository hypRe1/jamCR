from PIL import Image, ImageDraw, ImageFont

from asset_manager import AssetManager
from camera import Camera
from config import get_asset_dir

RESOLUTION = (540, 1200)

asset_manager = AssetManager(
    get_asset_dir()
)

camera = Camera(
    width=RESOLUTION[0],
    height=RESOLUTION[1],
    arena_left=11,
    arena_bottom=844,
    tile_width=28.7,
    tile_height=23.15,
)

arena = asset_manager.random_arena().convert("RGBA").copy()
draw = ImageDraw.Draw(arena)
font = ImageFont.load_default()


for x in range(19):
    p1 = camera.world_to_screen(x, 0)
    p2 = camera.world_to_screen(x, 32)
    draw.line([p1, p2], fill=(255, 255, 0), width=1)

for y in range(33):
    p1 = camera.world_to_screen(0, y)
    p2 = camera.world_to_screen(18, y)
    draw.line([p1, p2], fill=(255, 255, 0), width=1)


for x in range(18):
    for y in range(32):
        sx, sy = camera.world_to_screen(
            x + 0.5,
            y + 0.5,
        )

        # r = 2
        # draw.ellipse(
        #     (sx - r, sy - r, sx + r, sy + r),
        #     fill=(255, 0, 0),
        # )


for x in range(18):
    sx, sy = camera.world_to_screen(x + 0.5, 0.5)

    draw.text(
        (sx - 4, sy + 4),
        str(x),
        fill=(255, 255, 255),
        font=font,
    )


for y in range(32):
    sx, sy = camera.world_to_screen(0.5, y + 0.5)

    draw.text(
        (sx + 4, sy - 4),
        str(y),
        fill=(255, 255, 255),
        font=font,
    )


arena.save("coordinate_debug.png")

print("Saved coordinate_debug.png")
