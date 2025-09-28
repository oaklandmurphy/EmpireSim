import json
from OpenGL.GL import *
from OpenGL.GLU import *

import pygame
import random
import math
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



def display_square_grid(squaregrid, cell_size=40):
	"""
	Display the square grid in a separate pygame window.
	Each cell is drawn as a square. Cell color can be customized by adding a 'color' variable to cell variables.
	"""
	pygame.init()
	info = pygame.display.Info()
	screen_width = info.current_w
	screen_height = info.current_h
	screen = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
	pygame.display.set_caption("Square Grid Visualization")
	glViewport(0, 0, screen_width, screen_height)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluOrtho2D(0, screen_width, screen_height, 0)
	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()

	# Camera offset for panning
	cam_x, cam_y = 0, 0
	pan_speed = 40
	dragging = False
	last_mouse_pos = None

	def square_to_pixel(x, y):
		px = x * cell_size + cell_size - cam_x
		py = y * cell_size + cell_size - cam_y
		return int(px), int(py)

	def draw_square_opengl(x, y, size, color):
		glColor3ub(*color)
		glBegin(GL_QUADS)
		glVertex2f(x - size // 2, y - size // 2)
		glVertex2f(x + size // 2, y - size // 2)
		glVertex2f(x + size // 2, y + size // 2)
		glVertex2f(x - size // 2, y + size // 2)
		glEnd()

	# UI Button setup
	attributes = ['influence', 'solidarity', 'terrain', 'fitness', 'nation']
	selected_attributes = set(['terrain'])
	button_height = 40
	button_padding = 10
	button_font = pygame.font.SysFont('Arial', 24)

	recalc_button_label = 'Update'
	recalc_button_width = 160
	recalc_button_height = button_height
	continuous_button_label = 'Continuous'
	continuous_button_width = 180
	continuous_button_height = button_height
	continuous_update = False

	# Save button setup
	save_button_label = 'Save'
	save_button_width = 120
	save_button_height = button_height
	save_button_x = button_padding
	save_button_y = button_padding

	running = True
	clock = pygame.time.Clock()
	square_draw_size = int(cell_size * 0.95)

	# Tooltip setup
	tooltip_font = pygame.font.SysFont('Arial', 18)
	tooltip_bg = (30, 30, 30)
	tooltip_fg = (255, 255, 255)
	hovered_tile = None

	while running:
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT:
					cam_x -= pan_speed
				elif event.key == pygame.K_RIGHT:
					cam_x += pan_speed
				elif event.key == pygame.K_UP:
					cam_y -= pan_speed
				elif event.key == pygame.K_DOWN:
					cam_y += pan_speed
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 1:
					mouse_x, mouse_y = event.pos
					button_y = screen_height - button_height - button_padding
					# Check attribute buttons
					attr_clicked = False
					for i, attr in enumerate(attributes):
						button_width = screen_width // len(attributes) - button_padding
						button_x = i * (button_width + button_padding) + button_padding
						if button_x <= mouse_x <= button_x + button_width and button_y <= mouse_y <= button_y + button_height:
							# Toggle selection
							if attr in selected_attributes:
								selected_attributes.remove(attr)
							else:
								selected_attributes.add(attr)
							attr_clicked = True
							break
					# Check recalc button
					recalc_x = screen_width - recalc_button_width - button_padding - continuous_button_width - button_padding
					recalc_y = button_padding
					if recalc_x <= mouse_x <= recalc_x + recalc_button_width and recalc_y <= mouse_y <= recalc_y + recalc_button_height:
						if hasattr(squaregrid, 'calc_influence'):
							squaregrid.update_world()
						attr_clicked = True
					# Check continuous button
					continuous_x = screen_width - continuous_button_width - button_padding
					continuous_y = button_padding
					if continuous_x <= mouse_x <= continuous_x + continuous_button_width and continuous_y <= mouse_y <= continuous_y + continuous_button_height:
						continuous_update = not continuous_update
						attr_clicked = True
					# Check save button (top left)
					if save_button_x <= mouse_x <= save_button_x + save_button_width and save_button_y <= mouse_y <= save_button_y + save_button_height:
						import datetime
						timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
						filename = f'saved_maps/saved_map_{timestamp}.json'
						save_world_to_json(squaregrid, filename)
						print(f'Map saved to {filename}')
						attr_clicked = True
					# Otherwise, start dragging for panning
					if not attr_clicked:
						dragging = True
						last_mouse_pos = event.pos
			elif event.type == pygame.MOUSEBUTTONUP:
				if event.button == 1:
					dragging = False
			elif event.type == pygame.MOUSEMOTION:
				if dragging and last_mouse_pos:
					dx = event.pos[0] - last_mouse_pos[0]
					dy = event.pos[1] - last_mouse_pos[1]
					cam_x -= dx
					cam_y -= dy
					last_mouse_pos = event.pos
				else:
					# Tooltip: find hovered tile
					mouse_x, mouse_y = event.pos
					hovered_tile = None
					for x in range(squaregrid.width):
						for y in range(squaregrid.height):
							px, py = square_to_pixel(x, y)
							if (px - cell_size//2 <= mouse_x <= px + cell_size//2 and
								py - cell_size//2 <= mouse_y <= py + cell_size//2):
								hovered_tile = (x, y)
								break
						if hovered_tile:
							break
		# If continuous update is enabled, update the world every frame
		if continuous_update:
			if hasattr(squaregrid, 'update_world'):
				squaregrid.update_world()

		glClearColor(0.12, 0.12, 0.12, 1)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		
		for x in range(squaregrid.width):
			for y in range(squaregrid.height):
				px, py = square_to_pixel(x, y)
				if 0 - cell_size < px < screen_width + cell_size and 0 - cell_size < py < screen_height + cell_size:
					overlay_colors = []
					for attr in selected_attributes:
						minv = np.min(getattr(squaregrid, attr))
						maxv = np.max(getattr(squaregrid, attr))
						val = squaregrid.get_cell_var(x, y, attr)
						if maxv > minv:
							norm = (val - minv) / (maxv - minv)
						else:
							norm = 0.5
						if attr == 'terrain':
							green = int(220 - 120 * norm)
							color = (40, green, 40)
						elif attr == 'fitness':
							r = int(255 * (1 - norm))
							g = int(255 * norm)
							b = 40
							color = (r, g, b)
						elif attr == 'solidarity':
							r = int(255 * val)
							g = int(255 * (1 - val))
							b = 128
							color = (r, g, b)
						elif attr == 'nation':
							nation_id = int(val)
							color = squaregrid.nation_colors.get(nation_id, (100, 100, 100))
						else:
							if isinstance(val, (int, float)):
								color = (int(255 * norm), int(255 * (1 - norm)), 128)
							else:
								color = (200, 200, 200)
						overlay_colors.append(color)
					# Blend colors with alpha
					if overlay_colors:
						alpha = 1.0 / max(1, len(overlay_colors))
						r, g, b = 0, 0, 0
						for c in overlay_colors:
							r += c[0] * alpha
							g += c[1] * alpha
							b += c[2] * alpha
						color = (int(r), int(g), int(b))
					else:
						color = (200, 200, 200)
					draw_square_opengl(px, py, square_draw_size, color)

		# Draw UI buttons using OpenGL
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluOrtho2D(0, screen_width, screen_height, 0)
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()

		# Draw save button (top left) using OpenGL
		glColor3ub(60, 180, 60)
		glBegin(GL_QUADS)
		glVertex2f(save_button_x, save_button_y)
		glVertex2f(save_button_x + save_button_width, save_button_y)
		glVertex2f(save_button_x + save_button_width, save_button_y + save_button_height)
		glVertex2f(save_button_x, save_button_y + save_button_height)
		glEnd()
		glColor3ub(40, 40, 40)
		glBegin(GL_LINE_LOOP)
		glVertex2f(save_button_x, save_button_y)
		glVertex2f(save_button_x + save_button_width, save_button_y)
		glVertex2f(save_button_x + save_button_width, save_button_y + save_button_height)
		glVertex2f(save_button_x, save_button_y + save_button_height)
		glEnd()

		# Draw save button label using Pygame blit (like attribute buttons)
		# Draw save button label as OpenGL texture (fixes upside-down/distorted text)
		save_label = button_font.render(save_button_label, True, (255, 255, 255))
		save_label_width, save_label_height = save_label.get_size()
		save_label_data = pygame.image.tostring(save_label, "RGBA", True)
		save_texture_id = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, save_texture_id)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, save_label_width, save_label_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, save_label_data)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glEnable(GL_TEXTURE_2D)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glColor4f(1, 1, 1, 1)
		label_x = save_button_x + (save_button_width - save_label_width) // 2
		label_y = save_button_y + (save_button_height - save_label_height) // 2
		glBegin(GL_QUADS)
		glTexCoord2f(0, 1)
		glVertex2f(label_x, label_y)
		glTexCoord2f(1, 1)
		glVertex2f(label_x + save_label_width, label_y)
		glTexCoord2f(1, 0)
		glVertex2f(label_x + save_label_width, label_y + save_label_height)
		glTexCoord2f(0, 0)
		glVertex2f(label_x, label_y + save_label_height)
		glEnd()
		glDisable(GL_TEXTURE_2D)
		glDisable(GL_BLEND)
		glDeleteTextures([save_texture_id])

		for i, attr in enumerate(attributes):
			button_width = screen_width // len(attributes) - button_padding
			button_x = i * (button_width + button_padding) + button_padding
			button_y = screen_height - button_height - button_padding
			# Draw button rectangle
			if attr in selected_attributes:
				glColor3ub(100, 180, 100)
			else:
				glColor3ub(80, 80, 80)
			glBegin(GL_QUADS)
			glVertex2f(button_x, button_y)
			glVertex2f(button_x + button_width, button_y)
			glVertex2f(button_x + button_width, button_y + button_height)
			glVertex2f(button_x, button_y + button_height)
			glEnd()
			# Draw button border
			glColor3ub(40, 40, 40)
			glBegin(GL_LINE_LOOP)
			glVertex2f(button_x, button_y)
			glVertex2f(button_x + button_width, button_y)
			glVertex2f(button_x + button_width, button_y + button_height)
			glVertex2f(button_x, button_y + button_height)
			glEnd()


		# Draw recalc button
		recalc_x = screen_width - recalc_button_width - button_padding - continuous_button_width - button_padding
		recalc_y = button_padding
		glColor3ub(70, 120, 200)
		glBegin(GL_QUADS)
		glVertex2f(recalc_x, recalc_y)
		glVertex2f(recalc_x + recalc_button_width, recalc_y)
		glVertex2f(recalc_x + recalc_button_width, recalc_y + recalc_button_height)
		glVertex2f(recalc_x, recalc_y + recalc_button_height)
		glEnd()
		glColor3ub(40, 40, 40)
		glBegin(GL_LINE_LOOP)
		glVertex2f(recalc_x, recalc_y)
		glVertex2f(recalc_x + recalc_button_width, recalc_y)
		glVertex2f(recalc_x + recalc_button_width, recalc_y + recalc_button_height)
		glVertex2f(recalc_x, recalc_y + recalc_button_height)
		glEnd()

		# Draw continuous button
		continuous_x = screen_width - continuous_button_width - button_padding
		continuous_y = button_padding
		if continuous_update:
			glColor3ub(100, 200, 100)
		else:
			glColor3ub(120, 120, 120)
		glBegin(GL_QUADS)
		glVertex2f(continuous_x, continuous_y)
		glVertex2f(continuous_x + continuous_button_width, continuous_y)
		glVertex2f(continuous_x + continuous_button_width, continuous_y + continuous_button_height)
		glVertex2f(continuous_x, continuous_y + continuous_button_height)
		glEnd()
		glColor3ub(40, 40, 40)
		glBegin(GL_LINE_LOOP)
		glVertex2f(continuous_x, continuous_y)
		glVertex2f(continuous_x + continuous_button_width, continuous_y)
		glVertex2f(continuous_x + continuous_button_width, continuous_y + continuous_button_height)
		glVertex2f(continuous_x, continuous_y + continuous_button_height)
		glEnd()

		glMatrixMode(GL_MODELVIEW)
		glPopMatrix()
		glMatrixMode(GL_PROJECTION)
		glPopMatrix()

		# Draw button labels using Pygame font as OpenGL textures
		for i, attr in enumerate(attributes):
			button_width = screen_width // len(attributes) - button_padding
			button_x = i * (button_width + button_padding) + button_padding
			button_y = screen_height - button_height - button_padding
			label = button_font.render(attr.capitalize(), True, (255, 255, 255))
			label_data = pygame.image.tostring(label, "RGBA", True)
			label_width, label_height = label.get_size()
			texture_id = glGenTextures(1)
			glBindTexture(GL_TEXTURE_2D, texture_id)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, label_width, label_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, label_data)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glEnable(GL_TEXTURE_2D)
			glEnable(GL_BLEND)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
			center_x = button_x + button_width // 2
			center_y = button_y + button_height // 2
			glBindTexture(GL_TEXTURE_2D, texture_id)
			glColor4f(1, 1, 1, 1)
			glBegin(GL_QUADS)
			glTexCoord2f(0, 1)
			glVertex2f(center_x - label_width // 2, center_y - label_height // 2)
			glTexCoord2f(1, 1)
			glVertex2f(center_x + label_width // 2, center_y - label_height // 2)
			glTexCoord2f(1, 0)
			glVertex2f(center_x + label_width // 2, center_y + label_height // 2)
			glTexCoord2f(0, 0)
			glVertex2f(center_x - label_width // 2, center_y + label_height // 2)
			glEnd()
			glDisable(GL_TEXTURE_2D)
			glDisable(GL_BLEND)
			glDeleteTextures([texture_id])

		# Draw recalc button label
		recalc_label = button_font.render(recalc_button_label, True, (255, 255, 255))
		recalc_label_data = pygame.image.tostring(recalc_label, "RGBA", True)
		recalc_label_width, recalc_label_height = recalc_label.get_size()
		recalc_texture_id = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, recalc_texture_id)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, recalc_label_width, recalc_label_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, recalc_label_data)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glEnable(GL_TEXTURE_2D)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		recalc_center_x = recalc_x + recalc_button_width // 2
		recalc_center_y = recalc_y + recalc_button_height // 2
		glBindTexture(GL_TEXTURE_2D, recalc_texture_id)
		glColor4f(1, 1, 1, 1)
		glBegin(GL_QUADS)
		glTexCoord2f(0, 1)
		glVertex2f(recalc_center_x - recalc_label_width // 2, recalc_center_y - recalc_label_height // 2)
		glTexCoord2f(1, 1)
		glVertex2f(recalc_center_x + recalc_label_width // 2, recalc_center_y - recalc_label_height // 2)
		glTexCoord2f(1, 0)
		glVertex2f(recalc_center_x + recalc_label_width // 2, recalc_center_y + recalc_label_height // 2)
		glTexCoord2f(0, 0)
		glVertex2f(recalc_center_x - recalc_label_width // 2, recalc_center_y + recalc_label_height // 2)
		glEnd()
		glDisable(GL_TEXTURE_2D)
		glDisable(GL_BLEND)
		glDeleteTextures([recalc_texture_id])

		# Draw continuous button label
		continuous_label = button_font.render(continuous_button_label, True, (255, 255, 255))
		continuous_label_data = pygame.image.tostring(continuous_label, "RGBA", True)
		continuous_label_width, continuous_label_height = continuous_label.get_size()
		continuous_texture_id = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, continuous_texture_id)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, continuous_label_width, continuous_label_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, continuous_label_data)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glEnable(GL_TEXTURE_2D)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		continuous_center_x = continuous_x + continuous_button_width // 2
		continuous_center_y = continuous_y + continuous_button_height // 2
		glBindTexture(GL_TEXTURE_2D, continuous_texture_id)
		glColor4f(1, 1, 1, 1)
		glBegin(GL_QUADS)
		glTexCoord2f(0, 1)
		glVertex2f(continuous_center_x - continuous_label_width // 2, continuous_center_y - continuous_label_height // 2)
		glTexCoord2f(1, 1)
		glVertex2f(continuous_center_x + continuous_label_width // 2, continuous_center_y - continuous_label_height // 2)
		glTexCoord2f(1, 0)
		glVertex2f(continuous_center_x + continuous_label_width // 2, continuous_center_y + continuous_label_height // 2)
		glTexCoord2f(0, 0)
		glVertex2f(continuous_center_x - continuous_label_width // 2, continuous_center_y + continuous_label_height // 2)
		glEnd()
		glDisable(GL_TEXTURE_2D)
		glDisable(GL_BLEND)
		glDeleteTextures([continuous_texture_id])


		# Draw tooltip as OpenGL texture if hovering over a tile (before flip)
		if hovered_tile and not dragging:
			x, y = hovered_tile
			info = [
				f"x={x}, y={y}",
				f"influence: {squaregrid.influence[x, y]:.2f}",
				f"solidarity: {squaregrid.solidarity[x, y]:.2f}",
				f"terrain: {squaregrid.terrain[x, y]}",
				f"fitness: {squaregrid.fitness[x, y]:.2f}",
				f"nation: {squaregrid.nation[x, y]}"
			]
			surfaces = [tooltip_font.render(line, True, tooltip_fg, tooltip_bg) for line in info]
			width = max(s.get_width() for s in surfaces) + 10
			height = sum(s.get_height() for s in surfaces) + 10
			tooltip_img = pygame.Surface((width, height), pygame.SRCALPHA)
			tooltip_img.fill(tooltip_bg)
			y_offset = 5
			for s in surfaces:
				tooltip_img.blit(s, (5, y_offset))
				y_offset += s.get_height()
			mx, my = pygame.mouse.get_pos()
			tx = min(mx + 20, screen_width - width - 10)
			ty = min(my + 20, screen_height - height - 10)
			tooltip_data = pygame.image.tostring(tooltip_img, "RGBA", True)
			texture_id = glGenTextures(1)
			glBindTexture(GL_TEXTURE_2D, texture_id)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, tooltip_data)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glEnable(GL_TEXTURE_2D)
			glEnable(GL_BLEND)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
			glMatrixMode(GL_PROJECTION)
			glPushMatrix()
			glLoadIdentity()
			gluOrtho2D(0, screen_width, screen_height, 0)
			glMatrixMode(GL_MODELVIEW)
			glPushMatrix()
			glLoadIdentity()
			glBindTexture(GL_TEXTURE_2D, texture_id)
			glColor4f(1, 1, 1, 1)
			glBegin(GL_QUADS)
			glTexCoord2f(0, 1)
			glVertex2f(tx, ty)
			glTexCoord2f(1, 1)
			glVertex2f(tx + width, ty)
			glTexCoord2f(1, 0)
			glVertex2f(tx + width, ty + height)
			glTexCoord2f(0, 0)
			glVertex2f(tx, ty + height)
			glEnd()
			glDisable(GL_TEXTURE_2D)
			glDisable(GL_BLEND)
			glPopMatrix()
			glMatrixMode(GL_PROJECTION)
			glPopMatrix()
			glMatrixMode(GL_MODELVIEW)
			glDeleteTextures([texture_id])
		pygame.display.flip()
		clock.tick(60)
	pygame.quit()

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
	# Create a square grid

	world = load_world_from_json('saved_maps/80x50_basic.json')
	# world = World(80, 50)

	# Create nations in random locations

	coords = set()
	while len(coords) < 20:
		x = random.randint(0, world.width - 1)
		y = random.randint(0, world.height - 1)
		coords.add((x, y))
	for nation_id, (x, y) in enumerate(coords, start=1):
		world.create_nation(x, y, 10, 0.6)
		print(f"Created nation {nation_id} at ({x}, {y})")

	display_square_grid(world, cell_size=40)

if __name__ == "__main__":
	main()