import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT
import matplotlib.pyplot as plt

def main():
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT, 2)
    points = grid.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    # Save the plot to a file
    output_image = "scatter_plot.png"
    plt.savefig(output_image)
    print(f"Scatter plot saved as {output_image}")


def create_grid(width, height, ndim=1):
    grid_shape = (width, height, ndim)
    grid = (np.random.default_rng().random(size=(grid_shape))*2)-1
    return grid

if __name__ == "__main__":
    main()

