import numpy as np
from vector_grid import create_grid, draw_grid_vectors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from config import GRID_HEIGHT, GRID_WIDTH, NUM_POINTS


def main():
    FORCE_SCALE_FACTOR = 1.5
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT, 2)
    plot_filename = 'grid_vectors_t0.png'
    draw_grid_vectors(grid, plot_filename)
    # noise = get_2d_perlin_noise(grid, 3, 4)
    # print("noise value", noise)
    num_points = 1000
    # perlin_noise_grid = get_n_perlin_points(num_points, grid)

    # for i, x_y_magnitude in enumerate(perlin_noise_grid):
    #     x = x_y_magnitude[0]
    #     y = x_y_magnitude[1]
    #     z = x_y_magnitude[2]
    #     # print(i, f'  x: {x:.3f}   ',f'y: {y:.3f}   ', f'dot prod.: {z:.3f}')
    
    particles = []
    frames = []

    for _ in range(num_points):
        x = np.random.rand() * (GRID_WIDTH-1)
        y = np.random.rand() * (GRID_HEIGHT-1)
        particles.append(Particle(x, y))
    

    for t in range(50):
        new_points = {}
        for particle in particles:
            x, y = particle.position
            force_vector = get_force_at_point(x, y, grid)
            normalized_force_vector = normalize_vector(force_vector)
            force_vector = normalized_force_vector * FORCE_SCALE_FACTOR


            particle.apply_force(force_vector)
            particle.update()

        positions = np.array([p.position for p in particles])
        frames.append(positions)

    # Visualization setup
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, GRID_WIDTH)
    ax.set_ylim(0, GRID_HEIGHT)
    scatter = ax.scatter([], [], s=5, c='red', alpha=0.3)  # Scatter plot for points


    # for frame_idx, points_dict in enumerate(frames):
    #     print(f"Frame {frame_idx + 1}:")
    #     for position, force_vector in points_dict.items():
    #         print(f"  Position: {position}, Force vector: {force_vector}")

    


    def update(frame_index):
        """Update function for animation."""
        points_dict = frames[frame_index]
        # expects 2D array of shape (n, 2)
        positions = np.array(list(points_dict.keys()))
        scatter.set_offsets(positions)  # Update scatter plot positions
        return scatter,

        # Create the animation
    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)

    # Save the animation as a video file
    anim.save("flow_field_animation.mp4", writer="ffmpeg", fps=10)
    plt.close()

    print("Animation saved as flow_field_animation.mp4")

class Particle:
    def __init__(self, x, y):
        self.position = np.array([x, y])
        self.velocity = np.array([0, 0])
        self.acceleration = np.array([0, 0])

    def apply_force(self, force):
        self.acceleration += force

    def update(self, dt=1.0):
        self.velocity += self.acceleration * dt
        max_speed = 5
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity *= (max_speed / speed)

        self.position += self.velocity * dt

        # edges

        # Wrap around edges
        self.position[0] = self.position[0] % (GRID_WIDTH - 1)
        self.position[1] = self.position[1] % (GRID_HEIGHT - 1)
        
        # Reset acceleration for next frame
        # self.acceleration *= 0


def normalize_vector(vector):
    """Normalize a vector to unit length"""
    magnitude = np.linalg.norm(vector)
    # Avoid division by zero
    if magnitude < 1e-10:
        return np.array([1.0, 0.0])  # Return a default unit vector if magnitude is too small
    return vector / magnitude


def get_force_at_point(x, y, grid):
    width = len(grid[0])
    height = len(grid)
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # top/bottom, left/right of grid square
    vector_tl = grid[x0][y0]
    vector_tr = grid[x1][y0]
    vector_bl = grid[x0][y1]
    vector_br = grid[x1][y1]
    # print(vector_tl, vector_tr, vector_bl, vector_br)

    x_frac = x - x0
    y_frac = y - y0

    interp_top = vector_tl + x_frac * (vector_tr - vector_tl)
    interp_bottom = vector_bl + x_frac * (vector_br - vector_bl)

    result_vector = interp_top + y_frac * (interp_bottom - interp_top)
    return result_vector


def get_n_perlin_points(n, grid):
    grid_width, grid_height = grid.shape[0:2]

    points = []
    for _ in range(n):
        x = np.random.random() * (grid_width - 1)
        y = np.random.random() * (grid_height - 1)
        noise_value = get_2d_perlin_noise(grid, x, y)

        points.append((x, y, noise_value))

    return points


def get_2d_perlin_noise(gradient_grid, x, y):
    def fade(t):
        f_t = 6 * t**5 - 15 * t**4 + 10 * t**3
        return f_t

    def lerp(t, a0, a1):
        return a0 + t * (a1 - a0)

    grid_width, grid_height = gradient_grid.shape[:2]
    if x >= grid_width:
        raise ValueError("x value is larger than grid width")
    if y >= grid_width:
        raise ValueError("y value is larger than grid height")

    x_scaled = x * (grid_width - 1) / (grid_width - 1)
    y_scaled = y * (grid_height - 1) / (grid_height - 1)

    # get grid coords
    X = int(x_scaled)
    Y = int(y_scaled)

    x_frac = x_scaled - X
    y_frac = y_scaled - Y

    u = fade(x_frac)
    v = fade(y_frac)

    dp_at_x0 = np.dot(gradient_grid[X][Y], (0 - x, 0 - y))
    dp_at_x1 = np.dot(gradient_grid[X + 1][Y], (1 - x, 0 - y))
    dp_at_y0 = np.dot(gradient_grid[X][Y + 1], (0 - x, 1 - y))
    dp_at_y1 = np.dot(gradient_grid[X + 1][Y + 1], (1 - x, 1 - y))

    return lerp(v, lerp(u, dp_at_x0, dp_at_x1), lerp(u, dp_at_y0, dp_at_y1))


if __name__ == "__main__":
    main()
