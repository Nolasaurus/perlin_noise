import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT
import matplotlib.pyplot as plt


def main():
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT, 2)
    # plot_grid_vectors(grid)
    draw_grid_vectors(grid)


def create_grid(width, height, ndim=1):
    grid_shape = (width, height, ndim)
    grid = (np.random.default_rng().random(size=grid_shape) * 2) - 1
    return grid


def plot_grid_vectors(grid):
    # TODO: plot as arrows instead of just scatter
    points = grid.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    # Save the plot to a file
    output_image = "scatter_plot.png"
    plt.savefig(output_image)
    print(f"Scatter plot saved as {output_image}")


def draw_grid_vectors(grid):
    grid = get_unit_vectors(grid)
    x_components = grid[:, :, 0]
    y_components = grid[:, :, 1]

    rows, cols = grid.shape[:2]
    x_positions, y_positions = np.meshgrid(np.arange(cols), np.arange(rows))

    plt.figure(figsize=(8, 8))
    plt.quiver(
        x_positions,
        y_positions,
        x_components,
        y_components,
        angles="xy",
        scale_units="xy",
        scale=2,
    )
    plt.xticks(np.arange(0, cols))
    plt.yticks(np.arange(0, rows))
    plt.grid(
        True, which="both", linestyle="--", linewidth=0.5
    )  # Gridlines on all integers
    plt.savefig("grid_vectors.png")

    print("plot saved")


def get_unit_vectors(grid):
    # Extract the x and y components
    x_components = grid[:, :, 0]
    y_components = grid[:, :, 1]

    # Compute the magnitude of each vector
    magnitudes = np.sqrt(x_components**2 + y_components**2)

    # Avoid division by zero by replacing zeros with ones temporarily
    magnitudes[magnitudes == 0] = 1

    # Normalize x and y components to unit vectors
    x_components_normalized = x_components / magnitudes
    y_components_normalized = y_components / magnitudes

    unit_grid = np.stack([x_components_normalized, y_components_normalized], axis=-1)

    return unit_grid


if __name__ == "__main__":
    main()
