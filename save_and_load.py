from world import World
import numpy as np

def save_world_npz(grid, filename):
    np.savez_compressed(
        filename,
        width=grid.width,
        height=grid.height,
        influence=grid.influence,
        solidarity=grid.solidarity,
        terrain=grid.terrain,
        fitness=grid.fitness,
        nation=grid.nation,
        terrain_variability=getattr(grid, 'terrain_variablility', 5),
        num_nations=getattr(grid, 'num_nations', 0),
        nation_colors=np.array(
            [[k] + list(v) for k, v in getattr(grid, 'nation_colors', {}).items()]
        ),
        distance_matrix=getattr(grid, 'distance_matrix', None),
    )

def load_world_npz(filename):
    data = np.load(filename, allow_pickle=True)
    grid = World(int(data['width']), int(data['height']), data['distance_matrix'])
    grid.influence = data['influence']
    grid.solidarity = data['solidarity']
    grid.terrain = data['terrain']
    grid.fitness = data['fitness']
    grid.nation = data['nation']
    grid.terrain_variablility = int(data['terrain_variability'])
    grid.num_nations = int(data['num_nations'])
    # Convert nation_colors back
    grid.nation_colors = {
        int(row[0]): tuple(row[1:]) for row in data['nation_colors']
    }
    return grid