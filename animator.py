import numpy as np
import cv2 as cv

from config import TIME_STEPS, FRAME_WIDTH, FRAME_HEIGHT, DOT_COLOR, DOT_RADIUS, FPS, GRID_HEIGHT, GRID_WIDTH
from vector_grid import create_grid
from perlin_noise import get_2d_perlin_noise, get_n_perlin_points


def main():
    output_filename = 'drifting_random_gradient.mp4'
    fps = FPS
    ndim = 2
    grid = create_grid(GRID_WIDTH, GRID_HEIGHT, ndim)
    frame_height = FRAME_HEIGHT
    frame_width = FRAME_WIDTH
    frames = []

    frame = get_frame(frame_width, frame_height, grid)
    cv.imwrite('output_frame.png', frame)

    for _ in range(TIME_STEPS):
        new_frame = get_frame(frame_width, frame_height, grid)
        frames.append(new_frame)
        grid += 0.1 * create_grid(GRID_WIDTH, GRID_HEIGHT, ndim)
    
    get_video(output_filename, fps, frame_width, frame_height, frames)
    print(f'Video saved as {output_filename}')

def get_video(output_filename, fps, frame_width, frame_height, frames):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()


def get_frame(frame_width, frame_height, grid):
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


# def draw_line()




if __name__ == "__main__":
    main()