import random
import json
import numpy as np
from world import World
from opengl_draw import display_square_grid
from save_and_load import load_world_npz

def main():
    # Load or create a world
    world = load_world_npz('saved_maps/120x80_basic.npz')
    # world = World(120, 80)

    # Create nations in random locations
    coords = set()
    while len(coords) < 100:
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        coords.add((x, y))
    for nation_id, (x, y) in enumerate(coords, start=1):
        world.create_nation(x, y, 10, 0.6)
        print(f"Created nation {nation_id} at ({x}, {y})")

    display_square_grid(world, cell_size=40)

if __name__ == "__main__":
    main()


