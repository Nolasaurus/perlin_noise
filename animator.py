import numpy as np
import cv2 as cv
import matplotlib as mpl
from config import TIME_STEPS, FRAME_WIDTH, FRAME_HEIGHT, DOT_COLOR, DOT_RADIUS, FPS, GRID_HEIGHT, GRID_WIDTH, NUM_POINTS
from vector_grid import create_grid
from perlin_noise import get_2d_perlin_noise, get_n_perlin_points


def main():
    
    fps = FPS
    ndim = 2
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT, ndim)
    frame_height = FRAME_HEIGHT
    frame_width = FRAME_WIDTH
    frames = []
    perlin_frames = []

    frame = get_gradient_frame(frame_width, frame_height, grid)
    cv.imwrite('output_frame.png', frame)

    for _ in range(TIME_STEPS):
        new_frame = get_gradient_frame(frame_width, frame_height, grid)
        perlin_points = get_n_perlin_points(NUM_POINTS, grid)
        new_perlin_frame = get_perlin_frame(frame_width, frame_height, perlin_points)
        perlin_frames.append(new_perlin_frame)
        frames.append(new_frame)
        grid += 0.3 * create_grid(GRID_WIDTH, GRID_HEIGHT, ndim)

    
    output_filename = 'drifting_random_gradient.mp4'
    get_video(output_filename, fps, frame_width, frame_height, frames)
    output_filename = '2d_perlin_noise.mp4'
    get_video(output_filename, fps, frame_width, frame_height, perlin_frames)
    print(f'Video saved as {output_filename}')



def get_video(output_filename, fps, frame_width, frame_height, frames):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()


def get_gradient_frame(frame_width, frame_height, grid):
    frame = np.ones((frame_width, frame_height, 3), dtype=np.uint8) * 255
    grid_width = grid.shape[0]
    grid_height = grid.shape[1]
    grid = grid.copy().reshape(-1, 2)

    for ix, vector in enumerate(grid):
        i = ix % grid_width
        j = ix // grid_height

        # normalize vector
        magnitude = np.linalg.norm(vector)  # Compute the vector's magnitude
        if magnitude != 0:  # Avoid division by zero
            normalized_vector = vector / magnitude
        else:
            normalized_vector = vector

        scaled_vector = normalized_vector * 1
        start_point = (i * frame_width // grid_width, j * frame_height // grid_height)  # Starting point (scaled grid position)
        
        end_point = (int(start_point[0] + scaled_vector[0] * frame_width // grid_width),  # Ending point x
                    int(start_point[1] + scaled_vector[1] * frame_height // grid_height))  # Ending point y
        
        cv.line(frame, start_point, end_point, DOT_COLOR, DOT_RADIUS)
        cv.circle(frame, start_point, 2*DOT_RADIUS, DOT_COLOR)

    return frame


def get_perlin_frame(frame_width, frame_height, perlin_points):
    frame = np.ones((frame_width, frame_height, 3), dtype=np.uint8) * 255
    
    x_coords, y_coords, noise_values = perlin_points
    
    min_value = np.min(noise_values)
    max_value = np.max(noise_values)

    grid_width = np.max(x_coords)
    grid_height = np.max(y_coords)
    cmap = mpl.colormaps['viridis']

    for i, noise_value in enumerate(noise_values):
        # Normalize the noise value to [0, 1]
        normalized_value = (noise_value - min_value) / (max_value - min_value)

        # Get the RGBA color from the colormap
        rgba = cmap(normalized_value)

        # Convert RGBA to BGR and scale to 0-255
        bgr_color = tuple(int(c * 255) for c in rgba[:3][::-1])  # Reverse RGB to BGR
        x = x_coords[i]
        y = y_coords[i]
        
        # Draw the circle with the colormap color
        cv.circle(frame, (int(x * frame_width // grid_width), int(y * frame_height // grid_height)), 2 * DOT_RADIUS, bgr_color, -1)

    return frame

# def draw_line()




if __name__ == "__main__":
    main()