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
		self.solidarity = np.full((width, height), 0.5, dtype=float)
		self.terrain = np.full((width, height), 1, dtype=int)
		self.fitness = np.zeros((width, height), dtype=float)
		self.nation = np.zeros((width, height), dtype=int)
		self.
		self.color = np.zeros((width, height, 3), dtype=int)

		self.terrain_variablility = 5

		# Generate Perlin noise for each cell
		self.generate_perlin_noise_on_squaregrid(scale=1, seed=42, attribute='terrain', variability=self.terrain_variablility)
		# Map noise to color for visualization
		for x in range(self.width):
			for y in range(self.height):
				noise_val = self.get_cell_var(x, y, 'terrain') / self.terrain_variablility
				color = (200, int(noise_val * 255), 200)
				self.set_cell_var(x, y, 'color', color)

		print(self.terrain)
		# Store A* distance matrix
		self.distance_matrix = self.calc_distance_matrix()

	def set_cell_var(self, x, y, key, value):
		if key == 'influence':
			self.influence[x, y] = value
		elif key == 'solidarity':
			self.solidarity[x, y] = value
		elif key == 'terrain':
			self.terrain[x, y] = value
		elif key == 'fitness':
			self.fitness[x, y] = value
		elif key == 'nation':
			self.nation[x, y] = value
		elif key == 'color':
			self.color[x, y] = value
		else:
			raise KeyError(f"Unknown key: {key}")

	def get_cell_var(self, x, y, key):
		if key == 'influence':
			return self.influence[x, y]
		elif key == 'solidarity':
			return self.solidarity[x, y]
		elif key == 'terrain':
			return self.terrain[x, y]
		elif key == 'fitness':
			return self.fitness[x, y]
		elif key == 'nation':
			return self.nation[x, y]
		elif key == 'color':
			return tuple(self.color[x, y])
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
		mask = (self.nation == source_nation)
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

		for x in range(self.width):
			for y in range(self.height):
				self.calc_fitness(x=x, y=y)

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

	running = True
	clock = pygame.time.Clock()
	square_draw_size = int(cell_size * 0.95)
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

		glClearColor(0.12, 0.12, 0.12, 1)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		font = pygame.font.SysFont('Arial', max(10, int(cell_size * 0.5)))
		text_texture_cache = {}
		for x in range(squaregrid.width):
			for y in range(squaregrid.height):
				px, py = square_to_pixel(x, y)
				if 0 - cell_size < px < screen_width + cell_size and 0 - cell_size < py < screen_height + cell_size:
					color = tuple(squaregrid.color[x, y]) if hasattr(squaregrid, 'color') else (200, 200, 200)
					draw_square_opengl(px, py, square_draw_size, color)
					# Draw influence value in the center of the square using cached OpenGL texture
					influence = squaregrid.influence[x, y]
					fitness = squaregrid.fitness[x, y]
					# Influence text
					cache_key_inf = (f"inf_{influence}", font.get_height())
					if cache_key_inf not in text_texture_cache:
						text_surface_inf = font.render(str(influence), True, (0, 0, 0), None)
						text_data_inf = pygame.image.tostring(text_surface_inf, "RGBA", True)
						text_width_inf, text_height_inf = text_surface_inf.get_size()
						texture_id_inf = glGenTextures(1)
						glBindTexture(GL_TEXTURE_2D, texture_id_inf)
						glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width_inf, text_height_inf, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data_inf)
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
						text_texture_cache[cache_key_inf] = (texture_id_inf, text_width_inf, text_height_inf)
					else:
						texture_id_inf, text_width_inf, text_height_inf = text_texture_cache[cache_key_inf]
					# Fitness text
					cache_key_fit = (f"fit_{fitness}", font.get_height())
					if cache_key_fit not in text_texture_cache:
						text_surface_fit = font.render(str(round(fitness, 2)), True, (0, 0, 128), None)
						text_data_fit = pygame.image.tostring(text_surface_fit, "RGBA", True)
						text_width_fit, text_height_fit = text_surface_fit.get_size()
						texture_id_fit = glGenTextures(1)
						glBindTexture(GL_TEXTURE_2D, texture_id_fit)
						glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width_fit, text_height_fit, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data_fit)
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
						text_texture_cache[cache_key_fit] = (texture_id_fit, text_width_fit, text_height_fit)
					else:
						texture_id_fit, text_width_fit, text_height_fit = text_texture_cache[cache_key_fit]
					# Draw influence (higher in square)
					glEnable(GL_TEXTURE_2D)
					glEnable(GL_BLEND)
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
					glBindTexture(GL_TEXTURE_2D, texture_id_inf)
					glColor4f(1, 1, 1, 1)
					glBegin(GL_QUADS)
					glTexCoord2f(0, 1)
					glVertex2f(px - text_width_inf // 2, py - square_draw_size // 4 - text_height_inf // 2)
					glTexCoord2f(1, 1)
					glVertex2f(px + text_width_inf // 2, py - square_draw_size // 4 - text_height_inf // 2)
					glTexCoord2f(1, 0)
					glVertex2f(px + text_width_inf // 2, py - square_draw_size // 4 + text_height_inf // 2)
					glTexCoord2f(0, 0)
					glVertex2f(px - text_width_inf // 2, py - square_draw_size // 4 + text_height_inf // 2)
					glEnd()
					# Draw fitness (lower in square)
					glBindTexture(GL_TEXTURE_2D, texture_id_fit)
					glColor4f(1, 1, 1, 1)
					glBegin(GL_QUADS)
					glTexCoord2f(0, 1)
					glVertex2f(px - text_width_fit // 2, py + square_draw_size // 4 - text_height_fit // 2)
					glTexCoord2f(1, 1)
					glVertex2f(px + text_width_fit // 2, py + square_draw_size // 4 - text_height_fit // 2)
					glTexCoord2f(1, 0)
					glVertex2f(px + text_width_fit // 2, py + square_draw_size // 4 + text_height_fit // 2)
					glTexCoord2f(0, 0)
					glVertex2f(px - text_width_fit // 2, py + square_draw_size // 4 + text_height_fit // 2)
					glEnd()
					glDisable(GL_TEXTURE_2D)
					glDisable(GL_BLEND)
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

	world.calc_fitness_of_world()

	display_square_grid(world, cell_size=40)

if __name__ == "__main__":
	main()