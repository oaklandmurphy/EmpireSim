import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

import random
import numpy as np
import datetime
from save_and_load import save_world_npz

import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

def display_square_grid(squaregrid, cell_size=40):
	# --- Slider setup ---
	slider_width = 30
	slider_height = 200
	slider_padding = 80  # Increased from 30 to move sliders left
	slider_labels = [
		("Conquest Difficulty", "conquest_difficulty", 1.1, 5.0),
		("Nation Stability", "nation_stability", 0.0, 1.0),
		("Solidarity Spread Rate", "solidarity_spread_rate", 0.0, 1.0),
		("Conquest Assimilation Rate", "conquest_assimilation_rate", 0.0, 1.0)
	]
	dragging_slider = None
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

	# Now that screen_width and screen_height are defined, set up slider_rects
	slider_rects = []
	for i, (label, attr, minv, maxv) in enumerate(slider_labels):
		x = screen_width - slider_width - slider_padding
		y = 100 + i * (slider_height + 60)
		slider_rects.append(pygame.Rect(x, y, slider_width, slider_height))

	# UI Button setup (moved up to ensure button_font is defined before use)
	attributes = ['nation', 'influence', 'solidarity', 'fitness', 'terrain']
	selected_attribute = 'terrain'
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

	# --- Pre-render static UI textures (labels, button texts) ---
	static_textures = {}
	def create_texture_from_surface(surf):
		tex_data = pygame.image.tostring(surf, "RGBA", True)
		tex_id = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, tex_id)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surf.get_width(), surf.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		return tex_id, surf.get_width(), surf.get_height()

	# Pre-render attribute button labels
	for attr in attributes:
		label = button_font.render(attr.capitalize(), True, (255, 255, 255))
		static_textures[f'attr_{attr}'] = create_texture_from_surface(label)
	# Pre-render recalc, continuous, and save button labels
	static_textures['recalc'] = create_texture_from_surface(button_font.render(recalc_button_label, True, (255,255,255)))
	static_textures['continuous'] = create_texture_from_surface(button_font.render(continuous_button_label, True, (255,255,255)))
	static_textures['save'] = create_texture_from_surface(button_font.render(save_button_label, True, (255,255,255)))
	# Pre-render slider labels
	for i, (label, attr, minv, maxv) in enumerate(slider_labels):
		static_textures[f'slider_label_{i}'] = create_texture_from_surface(button_font.render(label, True, (255,255,255)))
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

	# UI Button setup
	selected_attributes = set(['nation'])
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

	# --- Efficient OpenGL texture rendering for the grid ---
	grid_texture_id = glGenTextures(1)
	grid_texture_needs_update = True

	def get_grid_color_array():
		# Returns a (height, width, 3) uint8 array of RGB values for the selected attribute
		attr = selected_attribute
		arr = getattr(squaregrid, attr)
		arr = np.asarray(arr)
		print()
		if attr == 'terrain':
			minv, maxv = np.min(arr), np.max(arr)
			norm = (arr - minv) / (maxv - minv) if maxv > minv else np.zeros_like(arr)
			green = (220 - 120 * norm).astype(np.uint8)
			color = np.stack([np.full_like(green, 40), green, np.full_like(green, 40)], axis=-1)
		elif attr == 'fitness':
			minv, maxv = np.min(arr), np.max(arr)
			norm = (arr - minv) / (maxv - minv) if maxv > minv else np.zeros_like(arr)
			r = (255 * (1 - norm)).astype(np.uint8)
			g = (255 * norm).astype(np.uint8)
			b = np.full_like(r, 40)
			color = np.stack([r, g, b], axis=-1)
		elif attr == 'solidarity':
			minv, maxv = 0, 1
			norm = (arr - minv) / (maxv - minv) if maxv > minv else np.zeros_like(arr)
			r = (255 * (1 - norm)).astype(np.uint8)
			g = (255 * norm).astype(np.uint8)
			b = np.full_like(r, 40)
			color = np.stack([r, g, b], axis=-1)
		elif attr == 'nation':
			color = np.zeros((squaregrid.width, squaregrid.height, 3), dtype=np.uint8)
			for nation_id, col in squaregrid.nation_colors.items():
				mask = (squaregrid.nation == nation_id)
				color[mask] = col
			color = color
		elif attr == 'influence':
			minv, maxv = 0, 5
			norm = (arr - minv) / (maxv - minv) if maxv > minv else np.zeros_like(arr)
			r = (255 * norm).astype(np.uint8)
			g = (255 * norm).astype(np.uint8)
			b = (255 * norm).astype(np.uint8)
			color = np.stack([r, g, b], axis=-1)
		else:
			minv, maxv = np.min(arr), np.max(arr)
			norm = (arr - minv) / (maxv - minv) if maxv > minv else np.zeros_like(arr)
			color = np.stack([r, g, b], axis=-1)
		# Transpose to (height, width, 3) for OpenGL
		return np.transpose(color, (1, 0, 2))

	# Tooltip setup
	tooltip_font = pygame.font.SysFont('Arial', 18)
	tooltip_bg = (30, 30, 30)
	tooltip_fg = (255, 255, 255)
	hovered_tile = None

	while running:
		events = pygame.event.get()
		for event in events:
			# If attribute selection changes, mark grid texture for update
			if event.type == pygame.MOUSEBUTTONDOWN:
				if event.button == 1:
					mouse_x, mouse_y = event.pos
					button_y = screen_height - button_height - button_padding
					for i, attr in enumerate(attributes):
						button_width = screen_width // len(attributes) - button_padding
						button_x = i * (button_width + button_padding) + button_padding
						if button_x <= mouse_x <= button_x + button_width and button_y <= mouse_y <= button_y + button_height:
							if selected_attribute != attr:
								selected_attribute = attr
								grid_texture_needs_update = True
							break
			# If world data changes (e.g., continuous update, recalc), mark grid texture for update
			if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
				grid_texture_needs_update = True
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
					# Check sliders first
					for i, rect in enumerate(slider_rects):
						if rect.collidepoint(mouse_x, mouse_y):
							dragging_slider = i
							attr_clicked = True
							break
					if not dragging_slider is None:
						continue
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
						timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
						filename = f'saved_maps/saved_map_{timestamp}.npz'
						save_world_npz(squaregrid, filename)
						print(f'Map saved to {filename}')
						attr_clicked = True
					# Otherwise, start dragging for panning
					if not attr_clicked:
						dragging = True
						last_mouse_pos = event.pos
				elif event.button == 4:
					# Zoom in
					cam_x = cam_x + squaregrid.width
					cam_y = cam_y + squaregrid.width
					cell_size = min(200, cell_size + 2)
					
				elif event.button == 5:
					# Zoom out
					cam_x = cam_x - squaregrid.width
					cam_y = cam_y - squaregrid.width
					cell_size = max(2, cell_size - 2)
					
			elif event.type == pygame.MOUSEBUTTONUP:
				if event.button == 1:
					dragging = False
					dragging_slider = None
			elif event.type == pygame.MOUSEMOTION:
				if dragging_slider is not None:
					# Update slider value
					rect = slider_rects[dragging_slider]
					label, attr, minv, maxv = slider_labels[dragging_slider]
					rel_y = max(0, min(rect.height, event.pos[1] - rect.y))
					value = maxv - (rel_y / rect.height) * (maxv - minv)
					setattr(squaregrid, attr, value)
				elif dragging and last_mouse_pos:
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
							if (px - cell_size <= mouse_x <= px and
								py - cell_size <= mouse_y <= py):
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
		
		# --- Draw the grid as a single OpenGL texture ---
		if grid_texture_needs_update or continuous_update:
			grid_colors = get_grid_color_array()
			glBindTexture(GL_TEXTURE_2D, grid_texture_id)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, grid_colors.shape[1], grid_colors.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, grid_colors)
			grid_texture_needs_update = False

		# Compute pixel offset for panning
		offset_x = -cam_x
		offset_y = -cam_y

		# Draw the grid texture as a single quad
		glEnable(GL_TEXTURE_2D)
		glBindTexture(GL_TEXTURE_2D, grid_texture_id)
		glColor3f(1, 1, 1)
		glBegin(GL_QUADS)
		# Lower-left
		glTexCoord2f(0, 0)
		glVertex2f(offset_x, offset_y)
		# Lower-right
		glTexCoord2f(1, 0)
		glVertex2f(offset_x + squaregrid.width * cell_size, offset_y)
		# Upper-right
		glTexCoord2f(1, 1)
		glVertex2f(offset_x + squaregrid.width * cell_size, offset_y + squaregrid.height * cell_size)
		# Upper-left
		glTexCoord2f(0, 1)
		glVertex2f(offset_x, offset_y + squaregrid.height * cell_size)
		glEnd()
		glDisable(GL_TEXTURE_2D)


		# Draw UI buttons using OpenGL
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluOrtho2D(0, screen_width, screen_height, 0)
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()

		# Draw sliders using OpenGL (so they are visible in OpenGL mode)
		for i, rect in enumerate(slider_rects):
			label, attr, minv, maxv = slider_labels[i]
			# Draw slider background (vertical bar)
			glColor3ub(60, 60, 60)
			glBegin(GL_QUADS)
			glVertex2f(rect.x, rect.y)
			glVertex2f(rect.x + rect.width, rect.y)
			glVertex2f(rect.x + rect.width, rect.y + rect.height)
			glVertex2f(rect.x, rect.y + rect.height)
			glEnd()
			# Draw slider handle
			value = getattr(squaregrid, attr, minv)
			rel = int((maxv - value) / (maxv - minv) * rect.height)
			handle_y = rect.y + rel
			glColor3ub(180, 180, 80)
			glBegin(GL_QUADS)
			glVertex2f(rect.x, handle_y-4)
			glVertex2f(rect.x + rect.width, handle_y-4)
			glVertex2f(rect.x + rect.width, handle_y+4)
			glVertex2f(rect.x, handle_y+4)
			glEnd()
			# Draw label (pre-rendered texture)
			tex_id, w, h = static_textures[f'slider_label_{i}']
			x = rect.x - w - 10
			y = rect.y + rect.height//2 - h//2
			glEnable(GL_TEXTURE_2D)
			glEnable(GL_BLEND)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
			glBindTexture(GL_TEXTURE_2D, tex_id)
			glColor4f(1, 1, 1, 1)
			glBegin(GL_QUADS)
			glTexCoord2f(0, 1)
			glVertex2f(x, y)
			glTexCoord2f(1, 1)
			glVertex2f(x + w, y)
			glTexCoord2f(1, 0)
			glVertex2f(x + w, y + h)
			glTexCoord2f(0, 0)
			glVertex2f(x, y + h)
			glEnd()
			glDisable(GL_TEXTURE_2D)
			glDisable(GL_BLEND)
			# Draw value (dynamic, still rendered per-frame)
			value_surface = button_font.render(f"{value:.2f}", True, (255,255,255))
			value_tex_id, vw, vh = create_texture_from_surface(value_surface)
			vx = rect.x + rect.width + 10
			vy = rect.y + rect.height//2 - vh//2
			glEnable(GL_TEXTURE_2D)
			glEnable(GL_BLEND)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
			glBindTexture(GL_TEXTURE_2D, value_tex_id)
			glColor4f(1, 1, 1, 1)
			glBegin(GL_QUADS)
			glTexCoord2f(0, 1)
			glVertex2f(vx, vy)
			glTexCoord2f(1, 1)
			glVertex2f(vx + vw, vy)
			glTexCoord2f(1, 0)
			glVertex2f(vx + vw, vy + vh)
			glTexCoord2f(0, 0)
			glVertex2f(vx, vy + vh)
			glEnd()
			glDisable(GL_TEXTURE_2D)
			glDisable(GL_BLEND)
			glDeleteTextures([value_tex_id])

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

		# Draw save button label using pre-rendered texture
		tex_id, w, h = static_textures['save']
		label_x = save_button_x + (save_button_width - w) // 2
		label_y = save_button_y + (save_button_height - h) // 2
		glEnable(GL_TEXTURE_2D)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glBindTexture(GL_TEXTURE_2D, tex_id)
		glColor4f(1, 1, 1, 1)
		glBegin(GL_QUADS)
		glTexCoord2f(0, 1)
		glVertex2f(label_x, label_y)
		glTexCoord2f(1, 1)
		glVertex2f(label_x + w, label_y)
		glTexCoord2f(1, 0)
		glVertex2f(label_x + w, label_y + h)
		glTexCoord2f(0, 0)
		glVertex2f(label_x, label_y + h)
		glEnd()
		glDisable(GL_TEXTURE_2D)
		glDisable(GL_BLEND)

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

		# Draw button labels using pre-rendered textures
		for i, attr in enumerate(attributes):
			button_width = screen_width // len(attributes) - button_padding
			button_x = i * (button_width + button_padding) + button_padding
			button_y = screen_height - button_height - button_padding
			tex_id, w, h = static_textures[f'attr_{attr}']
			center_x = button_x + button_width // 2
			center_y = button_y + button_height // 2
			glEnable(GL_TEXTURE_2D)
			glEnable(GL_BLEND)
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
			glBindTexture(GL_TEXTURE_2D, tex_id)
			glColor4f(1, 1, 1, 1)
			glBegin(GL_QUADS)
			glTexCoord2f(0, 1)
			glVertex2f(center_x - w // 2, center_y - h // 2)
			glTexCoord2f(1, 1)
			glVertex2f(center_x + w // 2, center_y - h // 2)
			glTexCoord2f(1, 0)
			glVertex2f(center_x + w // 2, center_y + h // 2)
			glTexCoord2f(0, 0)
			glVertex2f(center_x - w // 2, center_y + h // 2)
			glEnd()
			glDisable(GL_TEXTURE_2D)
			glDisable(GL_BLEND)

		# Draw recalc button label using pre-rendered texture
		tex_id, w, h = static_textures['recalc']
		recalc_center_x = recalc_x + recalc_button_width // 2
		recalc_center_y = recalc_y + recalc_button_height // 2
		glEnable(GL_TEXTURE_2D)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glBindTexture(GL_TEXTURE_2D, tex_id)
		glColor4f(1, 1, 1, 1)
		glBegin(GL_QUADS)
		glTexCoord2f(0, 1)
		glVertex2f(recalc_center_x - w // 2, recalc_center_y - h // 2)
		glTexCoord2f(1, 1)
		glVertex2f(recalc_center_x + w // 2, recalc_center_y - h // 2)
		glTexCoord2f(1, 0)
		glVertex2f(recalc_center_x + w // 2, recalc_center_y + h // 2)
		glTexCoord2f(0, 0)
		glVertex2f(recalc_center_x - w // 2, recalc_center_y + h // 2)
		glEnd()
		glDisable(GL_TEXTURE_2D)
		glDisable(GL_BLEND)

		# Draw continuous button label using pre-rendered texture
		tex_id, w, h = static_textures['continuous']
		continuous_center_x = continuous_x + continuous_button_width // 2
		continuous_center_y = continuous_y + continuous_button_height // 2
		glEnable(GL_TEXTURE_2D)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glBindTexture(GL_TEXTURE_2D, tex_id)
		glColor4f(1, 1, 1, 1)
		glBegin(GL_QUADS)
		glTexCoord2f(0, 1)
		glVertex2f(continuous_center_x - w // 2, continuous_center_y - h // 2)
		glTexCoord2f(1, 1)
		glVertex2f(continuous_center_x + w // 2, continuous_center_y - h // 2)
		glTexCoord2f(1, 0)
		glVertex2f(continuous_center_x + w // 2, continuous_center_y + h // 2)
		glTexCoord2f(0, 0)
		glVertex2f(continuous_center_x - w // 2, continuous_center_y + h // 2)
		glEnd()
		glDisable(GL_TEXTURE_2D)
		glDisable(GL_BLEND)


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
