import random
import numpy as np
from perlin_noise import PerlinNoise
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra


class World:
	
	def __init__(self, width, height, distance_matrix=None):
		"""
		Initialize a square grid of given width and height using NumPy arrays.
		"""
		self.width = width
		self.height = height
		self.influence = np.zeros((width, height), dtype=float)
		self.solidarity = np.zeros((width, height), dtype=float)
		self.terrain = np.ones((width, height), dtype=int)
		self.fitness = np.full((width, height), 1, dtype=float)
		self.nation = np.zeros((width, height), dtype=int)

		self.terrain_variability = 10
		self.econ_growth_rate = 1.0
		self.num_nations = 0
		self.nation_colors = {0: (200, 200, 200)}  # 0: unassigned

		self.conquest_difficulty = 3  # Higher means harder to conquer
		self.solidarity_spread_rate = 0.25  # Chance to copy solidarity from most influential neighbor
		self.nation_stability = 0.75  # Higher means harder to rebel
		self.conquest_assimilation_rate = 0.5  # Chance to copy solidarity from conquering nation

		# Generate Perlin noise for each cell
		self.generate_perlin_noise_on_squaregrid(scale=2, seed=42, attribute='terrain', variability=self.terrain_variability)
		self.terrain = self.terrain**2

		if distance_matrix is not None:
			self.distance_matrix = distance_matrix
		else:
			self.distance_matrix = self.calc_distance_matrix()

	def set_cell_var(self, x, y, key, value):
		arr = getattr(self, key, None)
		if arr is not None and hasattr(arr, '__getitem__'):
			arr[x, y] = value
		else:
			raise KeyError(f"Unknown key: {key}")

	def get_cell_var(self, x, y, key):
		arr = getattr(self, key, None)
		if arr is not None and hasattr(arr, '__getitem__'):
			val = arr[x, y]
			# Return tuple for color arrays
			if key == 'color' and hasattr(val, '__iter__'):
				return tuple(val)
			return val
		else:
			raise KeyError(f"Unknown key: {key}")

	def neighbors(self, x, y):
		# Returns coordinates of neighboring cells within bounds (4 directions)
		directions = [
			(1, 0), (0, -1),
			(-1, 0), (0, 1)
		]
		result = []
		for dx, dy in directions:
			nx, ny = x + dx, y + dy
			if 0 <= nx < self.width and 0 <= ny < self.height:
				result.append((nx, ny))
		return result
	
	def generate_perlin_noise_on_squaregrid(self, scale=0.1, seed=0, attribute='noise', variability=10):
		"""
		Vectorized Perlin noise generation for the grid.
		"""
		noise = PerlinNoise(octaves=4, seed=seed)
		xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing='ij')
		px = xs * scale / self.width
		py = ys * scale / self.height
		def noise_func(x, y):
			return noise([x, y])
		vnoise = np.vectorize(noise_func)
		raw_values = vnoise(px, py)
		normalized = (raw_values + 1) / 2
		output = (normalized * (variability - 1)).astype(int) + 1
		arr = getattr(self, attribute, None)
		if arr is not None and hasattr(arr, '__getitem__'):
			arr[:, :] = output
		else:
			raise KeyError(f"Unknown key: {attribute}")

	def calc_distance_matrix(self):
		"""
		Computes all-pairs shortest paths on the grid using orthogonal neighbors.
		The cost to enter a cell is given by self.terrain[nx, ny].
		Returns a 4D array dist[x1, y1, x2, y2].
		"""
		h, w = self.nation.shape
		N = h * w  # total number of nodes

		# Map 2D coordinates to 1D index
		def idx(x, y):
			return x * w + y

		# Create sparse adjacency matrix
		graph = lil_matrix((N, N), dtype=float)

		for x in range(h):
			for y in range(w):
				for nx, ny in self.neighbors(x, y):
					# Cost to move from (x,y) to neighbor = terrain at neighbor
					graph[idx(x, y), idx(nx, ny)] = self.terrain[nx, ny]

		# Compute all-pairs shortest paths (Dijkstra)
		dist_flat = dijkstra(csgraph=graph, directed=True)

		# Reshape flat distances to 4D array
		dist_matrix = dist_flat.reshape(h, w, h, w)

		return dist_matrix


	# def calc_distance_matrix(self):
	# 	"""
	# 	Returns a 4D numpy array dist[x1, y1, x2, y2] with the minimum cost from (x1, y1) to (x2, y2) using Dijkstra's algorithm,
	# 	where terrain value is the cost to enter a cell.
	# 	"""
	# 	import heapq
	# 	dist = np.full((self.width, self.height, self.width, self.height), np.inf, dtype=float)
	# 	for x1 in range(self.width):
	# 		for y1 in range(self.height):
	# 			print(f"Calculating distances from ({x1}, {y1})")
	# 			# Dijkstra's algorithm from (x1, y1) to all other cells
	# 			visited = np.zeros((self.width, self.height), dtype=bool)
	# 			local_dist = np.full((self.width, self.height), np.inf, dtype=float)
	# 			local_dist[x1, y1] = 0.0
	# 			heap = [(0.0, (x1, y1))]
	# 			while heap:
	# 				curr_cost, (cx, cy) = heapq.heappop(heap)
	# 				if visited[cx, cy]:
	# 					continue
	# 				visited[cx, cy] = True
	# 				for nx, ny in self.neighbors(cx, cy):
	# 					new_cost = curr_cost + self.terrain[nx, ny]
	# 					if new_cost < local_dist[nx, ny]:
	# 						local_dist[nx, ny] = new_cost
	# 						heapq.heappush(heap, (new_cost, (nx, ny)))
	# 			# Store distances for all destinations from (x1, y1)
	# 			for x2 in range(self.width):
	# 				for y2 in range(self.height):
	# 					dist[x1, y1, x2, y2] = local_dist[x2, y2]
	# 	return dist

	def calc_fitness(self):
		self.fitness.fill(0.0)
		nations = np.unique(self.nation)
		nations = nations[nations != 0]  # skip 0

		for n in nations:
			mask = (self.nation == n)
			indices = np.argwhere(mask)
			if len(indices) == 0:
				continue

			# Extract influence and solidarity of source cells
			influence_vals = self.influence[mask]
			solidarity_vals = self.solidarity[mask]

			# Compute contributions from each source cell
			# distance_matrix has shape (width, height, width, height) ?
			# We'll assume distance_matrix[x, y] gives full distance map from (x, y)
			for idx, (x, y) in enumerate(indices):
				dist = self.distance_matrix[x, y].copy()
				dist[x, y] = 1  # avoid division by zero
				self.fitness[mask] += influence_vals[idx] * solidarity_vals[idx] / dist[mask]


	def calc_influence(self):
		"""
		Vectorized calculation of influence for each cell in the world.
		Orthogonal neighbors only. Exact behavior as original loops.
		"""
		h, w = self.nation.shape
		self.influence = np.zeros((h, w), dtype=float)

		# Self influence: only for nonzero nation
		mask = (self.nation != 0)
		self.influence[mask] += 1.0 * (1 - self.solidarity[mask])

		# Up neighbor (x-1, y)
		if h > 1:
			same_mask = (self.nation[1:, :] == self.nation[:-1, :])
			diff_mask = ~same_mask & (self.nation[1:, :] != 0)

			self.influence[1:, :][same_mask] += self.solidarity[:-1, :][same_mask]
			self.influence[1:, :][diff_mask] += self.solidarity[1:, :][diff_mask] / 4

			# Also handle contribution to the neighbor cell (mirroring the original loop)
			self.influence[:-1, :][same_mask] += self.solidarity[1:, :][same_mask]
			self.influence[:-1, :][diff_mask] += self.solidarity[:-1, :][diff_mask] / 4

		# Down neighbor (x+1, y) — already handled by up
		# Left neighbor (x, y-1)
		if w > 1:
			same_mask = (self.nation[:, 1:] == self.nation[:, :-1])
			diff_mask = ~same_mask & (self.nation[:, 1:] != 0)

			self.influence[:, 1:][same_mask] += self.solidarity[:, :-1][same_mask]
			self.influence[:, 1:][diff_mask] += self.solidarity[:, 1:][diff_mask] / 4

			self.influence[:, :-1][same_mask] += self.solidarity[:, 1:][same_mask]
			self.influence[:, :-1][diff_mask] += self.solidarity[:, :-1][diff_mask] / 4
	
	def conquer(self):
		new_nation = self.nation.copy()
		new_solidarity = self.solidarity.copy()
		w, h = self.height, self.width

		directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

		for dx, dy in directions:
			# Start with zeros
			f_shifted = np.zeros_like(self.fitness)
			n_shifted = np.zeros_like(self.nation)
			s_shifted = np.zeros_like(self.solidarity)

			# Determine valid source and target slices
			if dx == -1:
				f_shifted[:-1, :] = self.fitness[1:, :]
				n_shifted[:-1, :] = self.nation[1:, :]
				s_shifted[:-1, :] = self.solidarity[1:, :]
			elif dx == 1:
				f_shifted[1:, :] = self.fitness[:-1, :]
				n_shifted[1:, :] = self.nation[:-1, :]
				s_shifted[1:, :] = self.solidarity[:-1, :]
			elif dy == -1:
				f_shifted[:, :-1] = self.fitness[:, 1:]
				n_shifted[:, :-1] = self.nation[:, 1:]
				s_shifted[:, :-1] = self.solidarity[:, 1:]
			elif dy == 1:
				f_shifted[:, 1:] = self.fitness[:, :-1]
				n_shifted[:, 1:] = self.nation[:, :-1]
				s_shifted[:, 1:] = self.solidarity[:, :-1]

			# Conquest condition
			diff_nation = (self.nation != n_shifted) & (n_shifted != 0)
			stronger = f_shifted > self.fitness * self.conquest_difficulty
			mask = diff_nation & stronger

			# Apply updates
			new_nation[mask] = n_shifted[mask]

			rand_vals = np.random.rand(h, w)
			assimilation_mask = mask & (rand_vals < self.conquest_assimilation_rate)
			new_solidarity[assimilation_mask] = s_shifted[assimilation_mask]

		self.nation = new_nation
		self.solidarity = new_solidarity


	def update_solidarity(self):
		"""
		Vectorized update of solidarity:
		- Cells may adopt highest-influence same-nation neighbor's solidarity.
		- Random mutation (+/- 0.1) applied.
		- Rebellion checks applied based on rebel thresholds.
		"""
		h, w = self.nation.shape
		new_solidarity = self.solidarity.copy()
		
		rebel_threshold = self.get_all_nations_avg_solidarity()
		
		# --- Orthogonal neighbor influence ---
		# Up neighbor
		up_same = (self.nation[1:, :] == self.nation[:-1, :]) & (self.nation[:-1, :] != 0)
		up_infl = np.zeros_like(self.influence)
		up_infl[1:, :][up_same] = self.influence[:-1, :][up_same]

		# Down neighbor
		down_same = (self.nation[:-1, :] == self.nation[1:, :]) & (self.nation[1:, :] != 0)
		down_infl = np.zeros_like(self.influence)
		down_infl[:-1, :][down_same] = self.influence[1:, :][down_same]

		# Left neighbor
		left_same = (self.nation[:, 1:] == self.nation[:, :-1]) & (self.nation[:, :-1] != 0)
		left_infl = np.zeros_like(self.influence)
		left_infl[:, 1:][left_same] = self.influence[:, :-1][left_same]

		# Right neighbor
		right_same = (self.nation[:, :-1] == self.nation[:, 1:]) & (self.nation[:, 1:] != 0)
		right_infl = np.zeros_like(self.influence)
		right_infl[:, :-1][right_same] = self.influence[:, 1:][right_same]

		# Stack neighbors to find max influence
		neighbor_infl_stack = np.stack([up_infl, down_infl, left_infl, right_infl])
		max_neighbor_infl = np.max(neighbor_infl_stack, axis=0)

		# Cells that have higher-influence neighbor
		adopt_mask = (max_neighbor_infl > self.influence) & (self.nation != 0)
		rand_vals = np.random.rand(h, w)
		spread_mask = adopt_mask & (rand_vals < self.solidarity_spread_rate)

		# For cells in spread_mask, assign solidarity from the neighbor with max influence
		# Find which neighbor had the max influence
		argmax_idx = np.argmax(neighbor_infl_stack[:, :, :], axis=0)  # 0=up,1=down,2=left,3=right

		# Prepare shifted arrays to assign exact solidarity
		neighbor_solidarity = np.zeros_like(self.solidarity)
		# Up
		mask = (argmax_idx == 0) & spread_mask
		neighbor_solidarity[1:, :][mask[1:, :]] = self.solidarity[:-1, :][mask[1:, :]]
		# Down
		mask = (argmax_idx == 1) & spread_mask
		neighbor_solidarity[:-1, :][mask[:-1, :]] = self.solidarity[1:, :][mask[:-1, :]]
		# Left
		mask = (argmax_idx == 2) & spread_mask
		neighbor_solidarity[:, 1:][mask[:, 1:]] = self.solidarity[:, :-1][mask[:, 1:]]
		# Right
		mask = (argmax_idx == 3) & spread_mask
		neighbor_solidarity[:, :-1][mask[:, :-1]] = self.solidarity[:, 1:][mask[:, :-1]]

		# Assign the new solidarity
		new_solidarity[spread_mask] = neighbor_solidarity[spread_mask]

		# --- Random mutation ---
		mutate_vals = np.random.rand(h, w)
		new_solidarity[mutate_vals > 0.9] = np.minimum(1.0, new_solidarity[mutate_vals > 0.9] + 0.1)
		new_solidarity[mutate_vals < 0.1] = np.maximum(0.0, new_solidarity[mutate_vals < 0.1] - 0.1)

		# --- Rebellion checks ---
		nation_ids = self.nation
		rebellion_attempts = np.random.rand(h, w)
		for nation_id, threshold in rebel_threshold.items():
			if threshold > (1 - self.nation_stability):
				continue
			# Cells belonging to this nation
			mask = (nation_ids == nation_id)
			# Cells eligible for rebellion
			eligible = mask & (rebellion_attempts > threshold) & (rebellion_attempts < new_solidarity) & (np.random.rand(h, w) < 0.1)
			idxs = np.argwhere(eligible)
			for x, y in idxs:
				tyrant = self.nation[x, y]
				self.create_nation(x, y, initial_influence=10.0, initial_solidarity=new_solidarity[x, y])
				self.create_rebellion(x, y, threshold=0.4, tyrant_nation=tyrant)

		self.solidarity = new_solidarity	

	def create_nation (self, center_x, center_y, initial_influence=10.0, initial_solidarity=0.5):
		"""
		Creates a nation at the specified center coordinates with given initial influence and solidarity.
		"""
		if not (0 <= center_x < self.width and 0 <= center_y < self.height):
			raise ValueError("Center coordinates out of bounds")
		self.num_nations += 1
		self.nation[center_x, center_y] = self.num_nations
		self.influence[center_x, center_y] = initial_influence
		self.solidarity[center_x, center_y] = initial_solidarity
		# Assign a random color for this nation
		if self.num_nations not in self.nation_colors:
			self.nation_colors[self.num_nations] = tuple(random.randint(0, 255) for _ in range(3))


	def create_rebellion(self, x, y, threshold, tyrant_nation=0):
		"""
		Converts all tiles that are contiguous with (x, y) and have solidarity above the given threshold
		to the nation of (x, y). Uses BFS for flood fill.
		"""
		rebel_nation = self.nation[x, y]
		visited = set()
		queue = [(x, y)]
		while queue:
			cx, cy = queue.pop(0)
			if (cx, cy) in visited:
				continue
			visited.add((cx, cy))
			if self.solidarity[cx, cy] >= threshold:
				self.nation[cx, cy] = rebel_nation
				for nx, ny in self.neighbors(cx, cy):
					if (nx, ny) not in visited and self.solidarity[nx, ny] >= threshold and self.nation[nx, ny] == tyrant_nation:
						queue.append((nx, ny))

	def calc_avg_solidarity_by_nation(self, nation_id):
		"""
		Calculates the average fitness of all cells belonging to the specified nation.
		"""
		mask = (self.nation == nation_id) & (self.nation != 0)
		if np.any(mask):
			avg_solidarity = np.mean(self.solidarity[mask])
			return avg_solidarity
		else:
			return 0.0
	
	def get_all_nations_avg_solidarity(self):
		"""
		Returns a dict mapping nation_id to avg_solidarity for each nation present in the world.
		"""
		nation_ids = np.unique(self.nation)
		nation_ids = nation_ids[nation_ids != 0]  # Exclude 0 (no nation)
		result = {}
		for nation_id in nation_ids:
			avg = self.calc_avg_solidarity_by_nation(nation_id)
			result[nation_id] = avg
		return result

	def update_world(self):
		"""
		Updates the world state by recalculating influence, fitness, solidarity, and handling conquests.
		"""
		self.calc_influence()
		self.calc_fitness()
		self.update_solidarity()
		self.conquer()
