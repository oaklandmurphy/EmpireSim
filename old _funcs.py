# def calc_fitness(self, x, y):
# 		"""
# 		Adds the influence of cell (x, y) to the fitness of all other cells with the same nation value, weighted by inverse distance.
# 		"""
# 		source_nation = self.nation[x, y]
# 		influence_val = self.influence[x, y]
# 		solidarity_val = self.solidarity[x, y]
# 		mask = (self.nation == source_nation) & (self.nation != 0)
# 		# Use A* distances for all cells
# 		distances = self.distance_matrix[x, y].copy()
# 		# Avoid division by zero for self
# 		distances[x, y] = 1
# 		# Weight by inverse distance
# 		weights = np.zeros_like(distances)
# 		weights[mask] = 1.0 / distances[mask]
# 		self.fitness[mask] += influence_val * solidarity_val * weights[mask]

# def calc_fitness_of_world(self):
# 	"""
# 	Vectorized calculation of fitness for all cells in the world based on their influence and nation.
# 	"""
# 	self.fitness.fill(0.0)
# 	# For each cell, add its influence to all cells of the same nation, weighted by inverse distance
# 	for x in range(self.width):
# 		for y in range(self.height):
# 			source_nation = self.nation[x, y]
# 			if source_nation == 0:
# 				continue
# 			influence_val = self.influence[x, y]
# 			solidarity_val = self.solidarity[x, y]
# 			mask = (self.nation == source_nation) & (self.nation != 0)
# 			distances = self.distance_matrix[x, y].copy()
# 			distances[x, y] = 1
# 			weights = np.zeros_like(distances)
# 			weights[mask] = 1.0 / distances[mask]
# 			self.fitness[mask] += influence_val * solidarity_val * weights[mask]


# def calc_influence(self):
	# 	self.influence = np.zeros((self.width, self.height), dtype=float)
	# 	# Vectorized update for cells with nation != 0 (self influence part)
	# 	mask = (self.nation != 0)
	# 	self.influence[mask] += 1 * (1 - self.solidarity[mask])
	# 	# Neighbor influence is still done in a loop (nontrivial to vectorize due to neighbor structure)
	# 	for x in range(self.width):
	# 		for y in range(self.height):
	# 			if self.nation[x, y] == 0:
	# 				continue
	# 			neighbors = self.neighbors(x, y)
	# 			for nx, ny in neighbors:
	# 				if self.nation[nx, ny] == self.nation[x, y]:
	# 					self.influence[x, y] += self.solidarity[nx, ny]
	# 				else:
	# 					self.influence[x, y] += self.solidarity[x, y] / 4


# def conquer(self):
# 		for x in range(self.width):
# 			for y in range(self.height):
# 				neighbors = self.neighbors(x, y)
# 				max_fitness = self.fitness[x, y]
# 				new_nation = self.nation[x, y]
# 				new_solidarity = self.solidarity[x, y]
# 				for nx, ny in neighbors:
# 					if self.nation[x, y] != self.nation[nx,ny] and self.fitness[nx, ny] > max_fitness * self.conquest_difficulty:
# 						max_fitness = self.influence[nx, ny]
# 						new_nation = self.nation[nx, ny]
# 						new_solidarity = self.solidarity[nx, ny]
# 				self.nation[x, y] = new_nation
# 				if random.random() < self.conquest_assimilation_rate:
# 					self.solidarity[x,y] = new_solidarity

# def update_solidarity(self):

# 	rebel_threshold = self.get_all_nations_avg_solidarity()

# 	for x in range(self.width):
# 		for y in range(self.height):
# 			neighbors = self.neighbors(x, y)
# 			same_nation_neighbors = [ (nx, ny) for nx, ny in neighbors if self.nation[nx, ny] == self.nation[x, y] and self.nation[x, y] != 0 ]
# 			if same_nation_neighbors:
# 				# Find neighbor with highest influence
# 				max_infl_n, max_infl = None, self.influence[x, y]
# 				for nx, ny in same_nation_neighbors:
# 					infl = self.influence[nx, ny]
# 					if infl > max_infl:
# 						max_infl = infl
# 						max_infl_n = (nx, ny)
# 				if max_infl_n and random.random() < self.solidarity_spread_rate:
# 					self.solidarity[x, y] = self.solidarity[max_infl_n]
# 			mutate = random.random()
# 			if mutate > 0.9:
# 				self.solidarity[x, y] = min(1.0, self.solidarity[x, y] + 0.1)
# 			elif mutate < 0.1:
# 				self.solidarity[x, y] = max(0.0, self.solidarity[x, y] - 0.1)

# 			nation_id = self.nation[x, y]
# 			rebellion_attempt = random.random()
# 			if nation_id != 0 and nation_id in rebel_threshold and rebel_threshold[nation_id] <= (1 - self.nation_stability) and rebellion_attempt > rebel_threshold[nation_id] and rebellion_attempt < self.solidarity[x, y] and random.random() < 0.1:
# 				tyrant = self.nation[x, y]
# 				self.create_nation(x, y, initial_influence=10.0, initial_solidarity=self.solidarity[x, y])
# 				self.create_rebellion(x, y, threshold=0.4, tyrant_nation=tyrant)