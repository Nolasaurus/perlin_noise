import numpy as np
from vector_grid import create_grid
from config import GRID_HEIGHT, GRID_WIDTH, NUM_POINTS


def main():
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT)
    noise = get_2d_perlin_noise(grid, 1, 1)
    print(noise)

    perlin_noise_grid = get_n_perlin_points(9, grid)
    print('x_coords', perlin_noise_grid[0], '\n')
    print('y_coords', perlin_noise_grid[1], '\n')
    print('values', perlin_noise_grid[2], '\n')



def get_n_perlin_points(n, grid):
    grid_width, grid_height = grid.shape[0:2]

    x_coords = []
    y_coords = []
    noise_values = []

    for _ in range(n):
        x  = np.random.random() * (grid_width-1)
        y = np.random.random() * (grid_height-1)
        noise_value = get_2d_perlin_noise(grid, x, y)
        
        x_coords.append(x)
        y_coords.append(y)
        noise_values.append(noise_value)

    return x_coords, y_coords, noise_values
    

def get_2d_perlin_noise(gradient_grid, x, y):
    def fade(t):
        f_t = 6 * t**5 - 15 * t**4 + 10 * t**3
        return f_t

    def lerp(t, a0, a1):
        return a0 + t * (a1-a0)

    grid_width, grid_height = gradient_grid.shape[:2]
    if x >= grid_width:
        raise ValueError('x value is larger than grid width')
    if y >= grid_width:
        raise ValueError('y value is larger than grid height')

    x_scaled = x * (grid_width - 1) / (grid_width - 1)
    y_scaled = y * (grid_height - 1) / (grid_height - 1)


    # get grid coords
    X = int(x_scaled)
    Y = int(y_scaled)

    x_frac = x_scaled - X
    y_frac = y_scaled - Y

    u = fade(x_frac)
    v = fade(y_frac)
    
    dp_at_x0 = np.dot(gradient_grid[X][Y], (0-x, 0-y))
    dp_at_x1 = np.dot(gradient_grid[X+1][Y], (1-x, 0-y))
    dp_at_y0 = np.dot(gradient_grid[X][Y+1], (0-x, 1-y))
    dp_at_y1 = np.dot(gradient_grid[X+1][Y+1], (1-x, 1-y))

    return lerp(v, lerp(u, dp_at_x0, dp_at_x1),
                    lerp(u, dp_at_y0, dp_at_y1))



if __name__ == "__main__":
    main()