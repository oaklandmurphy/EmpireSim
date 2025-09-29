import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

import random
import numpy as np
import datetime
from save_and_load import save_world_to_json

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
        ("Solidarity Spread", "solidarity_spread_rate", 0.0, 1.0),
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
            # Draw label and value as OpenGL textures
            label_surface = button_font.render(label, True, (255,255,255))
            value_surface = button_font.render(f"{value:.2f}", True, (255,255,255))
            for surf, x, y in [
                (label_surface, rect.x - label_surface.get_width() - 10, rect.y + rect.height//2 - label_surface.get_height()//2),
                (value_surface, rect.x + rect.width + 10, rect.y + rect.height//2 - value_surface.get_height()//2)
            ]:
                tex_data = pygame.image.tostring(surf, "RGBA", True)
                tex_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surf.get_width(), surf.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glEnable(GL_TEXTURE_2D)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glColor4f(1, 1, 1, 1)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 1)
                glVertex2f(x, y)
                glTexCoord2f(1, 1)
                glVertex2f(x + surf.get_width(), y)
                glTexCoord2f(1, 0)
                glVertex2f(x + surf.get_width(), y + surf.get_height())
                glTexCoord2f(0, 0)
                glVertex2f(x, y + surf.get_height())
                glEnd()
                glDisable(GL_TEXTURE_2D)
                glDisable(GL_BLEND)
                glDeleteTextures([tex_id])

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
