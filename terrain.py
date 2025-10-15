import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import truncnorm
from perlin_noise import PerlinNoise

class PlateTectonics:
	def __init__(self, width, height, num_plates=8, seed=42, sea_level=0.55):
		self.width = width
		self.height = height
		self.num_plates = num_plates
		self.seed = seed
		np.random.seed(seed)

		self.elevation = np.zeros((height, width), dtype=float)
		self.plate_id = np.zeros((height, width), dtype=int)

		self.plate_motion = np.zeros((num_plates, 2), dtype=float)  # (dy, dx)
		self.plate_motion_map = np.zeros((height, width, 2), dtype=float)  # (dy, dx) per cell
		self.plate_rotation = np.zeros(num_plates, dtype=float)
		self.plate_buoyancy = np.zeros(num_plates, dtype=float)

		self.sea_level = sea_level  # Elevation threshold for sea level


		self._generate_plates()
		self._assign_plate_motion()
		self.generate_motion_map()
		self._generate_elevation()
		self.apply_plate_motion_to_elevation()
		self._normalize_elevation()

	def _generate_plates(self):
		"""Generate natural plate shapes via probabilistic growth and fill holes with random neighbors."""
		self.plate_id[:] = -1  # mark all cells as unassigned

		# Pick random seed points
		seeds = np.random.randint(0, [self.height, self.width], size=(self.num_plates, 2))
		frontier = []

		for pid, (y, x) in enumerate(seeds):
			self.plate_id[y, x] = pid
			frontier.append((y, x, pid))

		directions = [(-1,0),(1,0),(0,-1),(0,1)]

		while np.any(self.plate_id == -1):
			if not frontier:
				# Frontier empty but unassigned cells remain
				unassigned = np.argwhere(self.plate_id == -1)
				y, x = unassigned[np.random.randint(len(unassigned))]
				# Pick random neighbor plate to assign
				neighbors = [(y+dy, x+dx) for dy, dx in directions
							if 0 <= y+dy < self.height and 0 <= x+dx < self.width
							and self.plate_id[y+dy, x+dx] != -1]
				if neighbors:
					ny, nx = neighbors[np.random.randint(len(neighbors))]
					pid = self.plate_id[ny, nx]
				else:
					pid = np.random.randint(self.num_plates)
				self.plate_id[y, x] = pid
				frontier.append((y, x, pid))
				continue

			# Pick a random frontier cell
			idx = np.random.randint(len(frontier))
			y, x, pid = frontier.pop(idx)

			for dy, dx in directions:
				ny, nx = y + dy, x + dx
				if 0 <= ny < self.height and 0 <= nx < self.width and self.plate_id[ny, nx] == -1:
					if np.random.rand() < 0.8:  # probabilistic growth
						self.plate_id[ny, nx] = pid
						frontier.append((ny, nx, pid))

		# Assign random buoyancy
		self.plate_buoyancy = truncnorm.rvs(0, 1, loc=0.5, scale=0.45, size=self.num_plates)

	def _assign_plate_motion(self):
		for i in range(self.num_plates):
			angle = np.random.uniform(0, 2*np.pi)
			speed = np.random.uniform(0.1, 1.0)
			self.plate_motion[i] = [np.sin(angle)*speed, np.cos(angle)*speed]
			self.plate_rotation[i] = np.random.uniform(-0.02, 0.02)

	def generate_motion_map(self, smooth_sigma=3.0):
		"""
		Generate a (H, W, 2) motion map for all cells based on plate motion and rotation.
		Optionally smooth the motion map to produce larger, smoother mountains/valleys.

		Args:
			smooth_sigma: float, standard deviation for Gaussian smoothing
		"""
		H, W = self.height, self.width
		motion_map = np.zeros((H, W, 2), dtype=float)

		# Compute plate centers
		centers = np.zeros((self.num_plates, 2), dtype=float)  # (y, x)
		for pid in range(self.num_plates):
			coords = np.argwhere(self.plate_id == pid)
			centers[pid] = coords.mean(axis=0)

		# Assign motion to each plate
		for pid in range(self.num_plates):
			cy, cx = centers[pid]
			mask = (self.plate_id == pid)
			ys, xs = np.nonzero(mask)

			rel_y = ys - cy
			rel_x = xs - cx

			# Rotational component
			rot_y = -rel_x * self.plate_rotation[pid]
			rot_x =  rel_y * self.plate_rotation[pid]

			# Linear drift
			drift_y, drift_x = self.plate_motion[pid]

			# Total motion
			motion_map[ys, xs, 0] = drift_y + rot_y
			motion_map[ys, xs, 1] = drift_x + rot_x

		# --- Smooth the motion map ---
		motion_map[..., 0] = gaussian_filter(motion_map[..., 0], sigma=smooth_sigma)
		motion_map[..., 1] = gaussian_filter(motion_map[..., 1], sigma=smooth_sigma)

		self.plate_motion_map = motion_map
	
	def apply_plate_motion_to_elevation(self, strength=1.0):
		"""
		Apply tectonic motion to elevation so that both sides of a plate boundary
		rise when converging, and both lower when diverging.
		"""
		H, W = self.height, self.width
		dy = self.plate_motion_map[..., 0]
		dx = self.plate_motion_map[..., 1]

		delta = np.zeros((H, W), dtype=float)

		# Vertical neighbors
		rel_y = dy[1:, :] - dy[:-1, :]
		rel_x = dx[1:, :] - dx[:-1, :]

		# Relative speed magnitude along vertical
		rel_mag = np.sqrt(rel_y**2 + rel_x**2)

		# Determine direction: converging (-) vs diverging (+)
		# Use dot product with vertical vector (1,0)
		# For simplicity, converging = negative, diverging = positive
		vertical_dot = rel_y * 1 + rel_x * 0
		vertical_delta = -vertical_dot * rel_mag

		# Apply symmetrically
		delta[1:, :] += vertical_delta
		delta[:-1, :] += vertical_delta

		# Horizontal neighbors
		rel_y = dy[:, 1:] - dy[:, :-1]
		rel_x = dx[:, 1:] - dx[:, :-1]

		rel_mag = np.sqrt(rel_y**2 + rel_x**2)
		horizontal_dot = rel_y * 0 + rel_x * 1
		horizontal_delta = -horizontal_dot * rel_mag

		delta[:, 1:] += horizontal_delta
		delta[:, :-1] += horizontal_delta

		# Normalize delta to 0-1
		delta -= delta.min()
		delta /= delta.max()

		# Apply to elevation with strength
		self.elevation += strength * delta
	
	def _generate_elevation(self, noise_scale=0.05, octaves=4):
		"""
		Set elevation based on plate buoyancy + Perlin noise (no normalization).
		Args:
			noise_scale: float, controls frequency of the Perlin noise
			octaves: int, number of octaves for layered Perlin noise
		"""
		H, W = self.height, self.width

		# Base elevation from plate buoyancy
		self.elevation = self.plate_buoyancy[self.plate_id].astype(float)

		# Create grid coordinates
		ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

		# Initialize Perlin noise array
		noise = np.zeros((H, W), dtype=float)

		# Single PerlinNoise instance
		pn = PerlinNoise(octaves=1)

		freq = noise_scale
		amp = 1.0

		for _ in range(octaves):
			# Evaluate Perlin noise over the entire grid
			noise_layer = np.vectorize(lambda x, y: pn([x*freq, y*freq]))(xs, ys)
			noise += amp * noise_layer
			amp *= 0.5
			freq *= 2

		# Add Perlin noise directly to elevation
		self.elevation += noise



	def _normalize_elevation(self):
		"""Normalize elevation to 0–1."""
		min_elev = self.elevation.min()
		max_elev = self.elevation.max()
		self.elevation = (self.elevation - min_elev) / (max_elev - min_elev)


# Test
pt = PlateTectonics(100, 100, num_plates=10)
print("Elevation range:", pt.elevation.min(), pt.elevation.max())
