import random
import numpy as np
from perlin_noise import PerlinNoise


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

		self.terrain_variablility = 10
		self.econ_growth_rate = 1.0
		self.num_nations = 0
		self.nation_colors = {0: (200, 200, 200)}  # 0: unassigned

		# Generate Perlin noise for each cell
		self.generate_perlin_noise_on_squaregrid(scale=2, seed=42, attribute='terrain', variability=self.terrain_variablility)
		self.terrain = self.terrain**2 / self.terrain_variablility

		print(self.terrain)
		# Store A* distance matrix
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
		Generates Perlin noise values for each cell in the square grid and stores them as a variable.
		Output is normalized to integers between 1 and 'variability'.
		"""
		noise = PerlinNoise(octaves=4, seed=seed)
		for x in range(self.width):
			for y in range(self.height):
				px = x * scale
				py = y * scale
				raw_value = noise([px / self.width, py / self.height])
				# Normalize to [0, 1]
				normalized = (raw_value + 1) / 2
				# Scale to [1, variability] as integer
				output = int(normalized * (variability - 1)) + 1
				self.set_cell_var(x, y, attribute, output)

	def calc_distance_matrix(self):
		"""
		Returns a 4D numpy array dist[x1, y1, x2, y2] with the minimum cost from (x1, y1) to (x2, y2) using Dijkstra's algorithm,
		where terrain value is the cost to enter a cell.
		"""
		import heapq
		dist = np.full((self.width, self.height, self.width, self.height), np.inf, dtype=float)
		for x1 in range(self.width):
			for y1 in range(self.height):
				print(f"Calculating distances from ({x1}, {y1})")
				# Dijkstra's algorithm from (x1, y1) to all other cells
				visited = np.zeros((self.width, self.height), dtype=bool)
				local_dist = np.full((self.width, self.height), np.inf, dtype=float)
				local_dist[x1, y1] = 0.0
				heap = [(0.0, (x1, y1))]
				while heap:
					curr_cost, (cx, cy) = heapq.heappop(heap)
					if visited[cx, cy]:
						continue
					visited[cx, cy] = True
					for nx, ny in self.neighbors(cx, cy):
						new_cost = curr_cost + self.terrain[nx, ny]
						if new_cost < local_dist[nx, ny]:
							local_dist[nx, ny] = new_cost
							heapq.heappush(heap, (new_cost, (nx, ny)))
				# Store distances for all destinations from (x1, y1)
				for x2 in range(self.width):
					for y2 in range(self.height):
						dist[x1, y1, x2, y2] = local_dist[x2, y2]
		return dist

	def calc_fitness(self, x, y):
		"""
		Adds the influence of cell (x, y) to the fitness of all other cells with the same nation value, weighted by inverse distance.
		"""
		source_nation = self.nation[x, y]
		influence_val = self.influence[x, y]
		solidarity_val = self.solidarity[x, y]
		mask = (self.nation == source_nation) & (self.nation != 0)
		# Use A* distances for all cells
		distances = self.distance_matrix[x, y].copy()
		# Avoid division by zero for self
		distances[x, y] = 1
		# Weight by inverse distance
		weights = np.zeros_like(distances)
		weights[mask] = 1.0 / distances[mask]
		self.fitness[mask] += influence_val * solidarity_val * weights[mask]

	def calc_fitness_of_world(self):
		"""
		Calculates fitness for all cells in the world based on their influence and nation.
		"""
		self.fitness.fill(0.0)  # Reset fitness to base value
		for x in range(self.width):
			for y in range(self.height):

				self.calc_fitness(x=x, y=y)

	def calc_influence(self):
		self.influence = np.zeros((self.width, self.height), dtype=float)
		for x in range(self.width):
			for y in range(self.height):
				if self.nation[x, y] == 0:
					continue
				# rate = self.econ_growth_rate / max(self.influence[x, y], .5)
				rate = 1
				neighbors = self.neighbors(x, y)
				self.influence[x, y] += rate * (1 - self.get_cell_var(x, y, 'solidarity'))
				for nx, ny in neighbors:
					if self.nation[nx, ny] == self.nation[x, y]:
						self.influence[x, y] += rate * self.get_cell_var(nx, ny, 'solidarity')
					else:
						self.influence[x, y] += rate * self.get_cell_var(nx, ny, 'solidarity') / 4
	
	def conquer(self):
		for x in range(self.width):
			for y in range(self.height):
				neighbors = self.neighbors(x, y)
				max_fitness = self.fitness[x, y]
				new_nation = self.nation[x, y]
				new_solidarity = self.solidarity[x, y]
				conquered = False
				for nx, ny in neighbors:
					if self.nation[x, y] != self.nation[nx,ny] and self.fitness[nx, ny] > max_fitness * 2:
						max_fitness = self.influence[nx, ny]
						new_nation = self.nation[nx, ny]
						new_solidarity = self.solidarity[nx, ny]
						conquered = True
				self.nation[x, y] = new_nation
				if random.random() < 0.5:
					self.solidarity[x,y] = new_solidarity

	def update_solidarity(self):

		rebel_threshold = self.get_all_nations_avg_solidarity()

		for x in range(self.width):
			for y in range(self.height):
				neighbors = self.neighbors(x, y)
				same_nation_neighbors = [ (nx, ny) for nx, ny in neighbors if self.nation[nx, ny] == self.nation[x, y] and self.nation[x, y] != 0 ]
				if same_nation_neighbors:
					# Find neighbor with highest influence
					max_infl_n, max_infl = None, self.influence[x, y]
					for nx, ny in same_nation_neighbors:
						infl = self.influence[nx, ny]
						if infl > max_infl:
							max_infl = infl
							max_infl_n = (nx, ny)
					if max_infl_n and random.random() < 0.2:
						self.solidarity[x, y] = self.solidarity[max_infl_n]
				mutate = random.random()
				if mutate > 0.9:
					self.solidarity[x, y] = min(1.0, self.solidarity[x, y] + 0.1)
				elif mutate < 0.1:
					self.solidarity[x, y] = max(0.0, self.solidarity[x, y] - 0.1)

				nation_id = self.nation[x, y]
				rebellion_attempt = random.random()
				if nation_id != 0 and nation_id in rebel_threshold and rebel_threshold[nation_id] <= .2 and rebellion_attempt > rebel_threshold[nation_id] and rebellion_attempt < self.solidarity[x, y]:
					tyrant = self.nation[x, y]
					self.create_nation(x, y, initial_influence=10.0, initial_solidarity=1)
					self.create_rebellion(x, y, threshold=0.4, tyrant_nation=tyrant)
			


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
			self.nation_colors[self.num_nations] = tuple(random.randint(40, 220) for _ in range(3))


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
		self.calc_fitness_of_world()
		self.update_solidarity()
		self.conquer()



from opengl_draw import display_square_grid

