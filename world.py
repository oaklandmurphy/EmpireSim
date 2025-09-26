import json
from OpenGL.GL import *
from OpenGL.GLU import *

import pygame
import math
import numpy as np
from perlin_noise import PerlinNoise


class World:
	def __init__(self, width, height):
		"""
		Initialize a square grid of given width and height using NumPy arrays.
		"""
		self.width = width
		self.height = height
		self.influence = np.ones((width, height), dtype=float)
		self.solidarity = np.zeros((width, height), dtype=float)
		self.terrain = np.ones((width, height), dtype=int)
		self.fitness = np.full((width, height), 1, dtype=float)
		self.nation = np.zeros((width, height), dtype=int)

		self.terrain_variablility = 5
		self.econ_growth_rate = 1.0

		# Generate Perlin noise for each cell
		self.generate_perlin_noise_on_squaregrid(scale=2, seed=42, attribute='terrain', variability=self.terrain_variablility)

		print(self.terrain)
		# Store A* distance matrix
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
		mask = (self.nation == source_nation) & (self.nation != 0)
		# Use A* distances for all cells
		distances = self.distance_matrix[x, y].copy()
		# Avoid division by zero for self
		distances[x, y] = 1
		# Weight by inverse distance
		weights = np.zeros_like(distances)
		weights[mask] = 1.0 / distances[mask]
		self.fitness[mask] += influence_val * weights[mask]

	def calc_fitness_of_world(self):
		"""
		Calculates fitness for all cells in the world based on their influence and nation.
		"""
		self.fitness.fill(1.0)  # Reset fitness to base value
		for x in range(self.width):
			for y in range(self.height):
				self.calc_fitness(x=x, y=y)

	def calc_influence(self):
		for x in range(self.width):
			for y in range(self.height):
				rate = self.econ_growth_rate / self.get_cell_var(x, y, 'influence')
				neighbors = self.neighbors(x, y)
				self.influence[x, y] += rate * (1 - self.get_cell_var(x, y, 'solidarity'))
				for nx, ny in neighbors:
					if self.nation[nx, ny] == self.nation[x, y]:
						self.influence[x, y] += rate * self.get_cell_var(nx, ny, 'solidarity')
	
	def conquer(self):
		for x in range(self.width):
			for y in range(self.height):
				neighbors = self.neighbors(x, y)
				max_fitness = self.fitness[x, y]
				new_nation = self.nation[x, y]
				for nx, ny in neighbors:
					if self.nation[x, y] != self.nation[nx,ny] and self.fitness[nx, ny] > max_fitness * 1.5:
						max_fitness = self.influence[nx, ny]
						new_nation = self.nation[nx, ny]
				self.nation[x, y] = new_nation

	def update_solidarity(self, delta=0.01):
		for x in range(self.width):
			for y in range(self.height):
				neighbors = self.neighbors(x, y)
				same_nation_count = sum(1 for nx, ny in neighbors if self.nation[nx, ny] == self.nation[x, y])
				if same_nation_count >= 3:
					self.solidarity[x, y] = min(1.0, self.solidarity[x, y] + delta)
				else:
					self.solidarity[x, y] = max(0.0, self.solidarity[x, y] - delta)


	def create_nation (self, nation_id, center_x, center_y, initial_influence=10.0, initial_solidarity=0.5):
		"""
		Creates a nation at the specified center coordinates with given initial influence and solidarity.
		"""
		if not (0 <= center_x < self.width and 0 <= center_y < self.height):
			raise ValueError("Center coordinates out of bounds")
		self.nation[center_x, center_y] = nation_id
		self.influence[center_x, center_y] = initial_influence
		self.solidarity[center_x, center_y] = initial_solidarity

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
	selected_attribute = 'terrain'
	button_height = 40
	button_padding = 10
	button_font = pygame.font.SysFont('Arial', 24)
	recalc_button_label = 'Update'
	recalc_button_width = 160
	recalc_button_height = button_height

	running = True
	clock = pygame.time.Clock()
	square_draw_size = int(cell_size * 0.95)

	# Tooltip setup
	tooltip_font = pygame.font.SysFont('Arial', 18)
	tooltip_bg = (30, 30, 30)
	tooltip_fg = (255, 255, 255)
	hovered_tile = None

	while running:
		for event in pygame.event.get():
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
							selected_attribute = attr
							attr_clicked = True
							break
					# Check recalc button
					recalc_x = screen_width - recalc_button_width - button_padding
					recalc_y = button_padding
					if recalc_x <= mouse_x <= recalc_x + recalc_button_width and recalc_y <= mouse_y <= recalc_y + recalc_button_height:
						if hasattr(squaregrid, 'calc_influence'):
							squaregrid.update_world()
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

		glClearColor(0.12, 0.12, 0.12, 1)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		
		for x in range(squaregrid.width):
			for y in range(squaregrid.height):
				px, py = square_to_pixel(x, y)
				if 0 - cell_size < px < screen_width + cell_size and 0 - cell_size < py < screen_height + cell_size:
					minv = np.min(getattr(squaregrid, selected_attribute))
					maxv = np.max(getattr(squaregrid, selected_attribute))
					val = squaregrid.get_cell_var(x, y, selected_attribute)
					# Color tiles based on selected attribute
					if maxv > minv:
						norm = (val - minv) / (maxv - minv)
					else:
						norm = 0.5
					if selected_attribute == 'terrain':                        
						# Darker green for higher values
						green = int(220 - 120 * norm)  # 220 (bright) to 100 (dark)
						color = (40, green, 40)
					elif selected_attribute == 'fitness':
						# Red to green gradient
						r = int(255 * (1 - norm))
						g = int(255 * norm)
						b = 40
						color = (r, g, b)
					elif selected_attribute == 'nation':
						# Assign distinct colors for each nation ID
						nation_colors = [
							(200, 200, 200), # 0: unassigned
							(220, 50, 50),   # 1: red
							(50, 220, 50),   # 2: green
							(50, 50, 220),   # 3: blue
							(220, 220, 50),  # 4: yellow
							(220, 50, 220),  # 5: magenta
							(50, 220, 220),  # 6: cyan
							(150, 80, 220),  # 7: purple
							(220, 120, 40),  # 8: orange
							(120, 220, 40),  # 9: lime
						]
						nation_id = int(val)
						if 0 <= nation_id < len(nation_colors):
							color = nation_colors[nation_id]
						else:
							color = (100, 100, 100)
					else:
						if isinstance(val, (int, float)):
							color = (int(255 * norm), int(255 * (1 - norm)), 128)
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

		for i, attr in enumerate(attributes):
			button_width = screen_width // len(attributes) - button_padding
			button_x = i * (button_width + button_padding) + button_padding
			button_y = screen_height - button_height - button_padding
			# Draw button rectangle
			if attr == selected_attribute:
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
		recalc_x = screen_width - recalc_button_width - button_padding
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
		'color': grid.color.tolist(),
		'terrain_variablility': getattr(grid, 'terrain_variablility', 5)
	}
	with open(filename, 'w') as f:
		json.dump(data, f)

def load_world_from_json(filename):
	"""
	Load a SquareGrid object from a JSON file.
	"""
	with open(filename, 'r') as f:
		data = json.load(f)
	grid = World(data['width'], data['height'])
	grid.influence = np.array(data['influence'], dtype=float)
	grid.solidarity = np.array(data['solidarity'], dtype=float)
	grid.terrain = np.array(data['terrain'], dtype=int)
	grid.fitness = np.array(data['fitness'], dtype=float)
	grid.nation = np.array(data['nation'], dtype=int)
	grid.color = np.array(data['color'], dtype=int)
	grid.terrain_variablility = data.get('terrain_variablility', 5)
	return grid

def main():
	# Create a square grid
	world = World(32, 32)

	# Create 5 nations in random locations
	import random
	coords = set()
	while len(coords) < 5:
		x = random.randint(0, world.width - 1)
		y = random.randint(0, world.height - 1)
		coords.add((x, y))
	for nation_id, (x, y) in enumerate(coords, start=1):
		world.create_nation(nation_id, x, y)
		print(f"Created nation {nation_id} at ({x}, {y})")

	display_square_grid(world, cell_size=40)

if __name__ == "__main__":
	main()