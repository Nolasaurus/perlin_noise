import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT

def main():
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT)

def create_grid(width, height):
    grid_shape = (width, height)
    grid = (np.random.default_rng(seed=22).random(size=(grid_shape))*2)-1
    return grid

if __name__ == "__main__":
    main()