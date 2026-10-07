from dataclasses import dataclass


@dataclass
class Camera:
    width: int
    height: int

    arena_left: float = 11
    arena_bottom: float = 844

    tile_width: float = 28.7
    tile_height: float = 23.15

    world_width: float = 18
    world_height: float = 32

    def world_to_screen(self, x, y):
        screen_x = self.arena_left + x * self.tile_width
        screen_y = self.arena_bottom - (y - 1) * self.tile_height

        return screen_x, screen_y

    def screen_to_world(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:

        dx = x - self.arena_left
        dy = self.arena_bottom - y

        world_x = dx / self.tile_width - dy / self.tile_height + 1

        world_y = dx / self.tile_width + dy / self.tile_height

        return world_x, world_y
