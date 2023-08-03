import pygame
from Doodle_trainer import Train,Data
from nn_3 import NeuralNetwork as nn



def get_grid_array(screen, grid_size, block_size):
    grid_array = []
    for y in range(grid_size):
        row = []
        for x in range(grid_size):
            color = screen.get_at((x * block_size, y * block_size))[:3]  # Get RGB values (ignore alpha)
            is_black = color == (0, 0, 0)     # Check if the color is black
            row.append(1 if is_black else 0)  # Add 1 for black, 0 for not black
        grid_array.append(row)
    return grid_array

def concatenate_arrays(grid_array):
    return [item for row in grid_array for item in row]
def main():
    pygame.init()

    # Set up the screen and grid parameters
    screen_size = (800, 800)
    grid_size = 28
    block_size = screen_size[0] // grid_size
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Draw on Grid")
    WHITE = (255, 255, 255)
    screen.fill(WHITE)

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle drawing on the grid
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left-click to draw
                    x, y = pygame.mouse.get_pos()
                    grid_x = x // block_size
                    grid_y = y // block_size
                    pygame.draw.rect(screen, (0, 0, 0), (grid_x * block_size, grid_y * block_size, block_size, block_size))

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # When spacebar is pressed, get the grid array and print it
                    grid_array = get_grid_array(screen, grid_size, block_size)
                    flat_array = concatenate_arrays(grid_array)
                    print(flat_array)

        pygame.display.flip()

    pygame.quit()





if __name__ == "__main__":
    nn = nn(784, [256, 128], 4)
    main()
