import random
import json
import numpy as np
from world import World
from opengl_draw import display_square_grid
from save_and_load import save_world_to_json, load_world_from_json

def save_world_to_json(grid, filename):
	"""
	Save a SquareGrid object to a JSON file.
	"""
	data = {
		'width': grid.width,
		'height': grid.height,
		'influence': grid.influence.tolist(),
		'solidarity': grid.solidarity.tolist(),
		'terrain': grid.terrain.tolist(),
		'fitness': grid.fitness.tolist(),
		'nation': grid.nation.tolist(),
		'terrain_variablility': getattr(grid, 'terrain_variablility', 5),
		'num_nations': getattr(grid, 'num_nations', 0),
		'nation_colors': {str(k): list(v) for k, v in getattr(grid, 'nation_colors', {}).items()},
		'distance_matrix': grid.distance_matrix.tolist() if hasattr(grid, 'distance_matrix') else [],
	}
	with open(filename, 'w') as f:
		json.dump(data, f)

def load_world_from_json(filename):
	"""
	Load a SquareGrid object from a JSON file.
	"""
	with open(filename, 'r') as f:
		data = json.load(f)
	grid = World(data['width'], data['height'], np.array(data['distance_matrix']))
	grid.influence = np.array(data['influence'], dtype=float)
	grid.solidarity = np.array(data['solidarity'], dtype=float)
	grid.terrain = np.array(data['terrain'], dtype=int)
	grid.fitness = np.array(data['fitness'], dtype=float)
	grid.nation = np.array(data['nation'], dtype=int)
	grid.terrain_variablility = data.get('terrain_variablility', 5)
	grid.num_nations = data.get('num_nations', 0)
	# Convert nation_colors back to int keys and tuple values
	nc = data.get('nation_colors', {})
	grid.nation_colors = {int(k): tuple(v) for k, v in nc.items()}
	return grid

def main():
    # Load or create a world
    world = load_world_from_json('saved_maps/80x50_basic.json')
    # world = World(48, 24)

    # Create nations in random locations
    coords = set()
    while len(coords) < 10:
        x = random.randint(0, world.width - 1)
        y = random.randint(0, world.height - 1)
        coords.add((x, y))
    for nation_id, (x, y) in enumerate(coords, start=1):
        world.create_nation(x, y, 10, 0.6)
        print(f"Created nation {nation_id} at ({x}, {y})")

    display_square_grid(world, cell_size=40)

if __name__ == "__main__":
    main()
