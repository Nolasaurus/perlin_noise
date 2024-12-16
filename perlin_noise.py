import numpy as np
from vector_grid import create_grid
from config import GRID_HEIGHT, GRID_WIDTH, NUM_POINTS


def main():
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT, 2)
    noise = get_2d_perlin_noise(grid, 3, 4)
    print("noise", noise)
    num_points = 9
    perlin_noise_grid = get_n_perlin_points(num_points, grid)

    for i, x_y_magnitude in enumerate(perlin_noise_grid):
        x = x_y_magnitude[0]
        y = x_y_magnitude[1]
        z = x_y_magnitude[2]
        # print(i, f'  x: {x:.3f}   ',f'y: {y:.3f}   ', f'dot prod.: {z:.3f}')
    force_vector = get_force_at_point(x, y, grid)
    print(force_vector)


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
    print(vector_tl, vector_tr, vector_bl, vector_br)

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
