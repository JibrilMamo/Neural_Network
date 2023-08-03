from nn_3 import NeuralNetwork as nn
import pygame
import numpy as np
from tqdm import tqdm
import random
import sys
from PIL import Image

pygame.init()

nn = nn(784,[256,128],4)
lable = ["Cat","Dog","Airplane","Smiley Face"]

width, height = 600, 600
window = pygame.display.set_mode((width, height))

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class Data:
    CAT = 0
    DOG = 1
    AIRPLANE = 2
    SMILE = 3

    def __init__(self, category, data):
        self.category = category
        self.data = data

    @staticmethod
    def load_data(max_img):
        print("Loading...")
        label_mapping = {
            'Data/cat.npy': Data.CAT,
            'Data/dog.npy': Data.DOG,
            'Data/airplane.npy': Data.AIRPLANE,
            'Data/smiley face.npy': Data.SMILE
        }

        data_obj = []

        for file_path, label in label_mapping.items():
            data = np.load(file_path)
            num_images_to_load = min(max_img, len(data))
            for item in data[:num_images_to_load]:
                data_obj.append(Data(label, item / 255))

        print("Finished Collecting")

        np.random.shuffle(data_obj)
        split_index = int(0.8 * len(data_obj))

        # Split the array into two parts: 80% and the remaining 20%
        data = (data_obj[:split_index], data_obj[split_index:])

        return data


class Train:
    def __init__(self, nn, lr):
        self.NN = nn
        self.learning_rate = lr
        self.NN.learning_rate = self.learning_rate

    def train(self, data, test_data, epoch=50):
        print("Started")
        self.test(test_data)
        for j in tqdm(range(epoch)):
            np.random.shuffle(data)
            for i, item in enumerate(data):
                target = [0, 0, 0, 0]
                target[item.category] = 1
                self.NN.train(item.data, target)

            if j % (epoch // 5) == 0:
                np.random.shuffle(test_data)
                self.test(test_data)
        self.test(test_data)

    def test(self, data):
        result = sum(1 for item in data if self.get_predicted_category(item) == item.category)
        acc = result / len(data)


        print('\n')
        ind = random.randint(1,4)
        print(self.NN.feedforward(data[(len(data)-1)//ind].data))
        target = [0,0,0,0]
        target[data[(len(data)-1)//ind].category] = 1
        print(target)
        print("Accuracy: ", acc)
        print('\n')

    def get_predicted_category(self, item):
        output = self.NN.feedforward(item.data)
        return max(range(len(output)), key=lambda i: output[i])


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
def display_text(window, text, fontsize, x, y):
    pygame.font.init()
    font = pygame.font.SysFont(None, fontsize)
    text_surface = font.render(text, True, (0, 0, 0))  # White color (R,G,B)
    window.fill((255,255,255), (x- text_surface.get_width(), y, window.get_width(), text_surface.get_height()+10))
    window.blit(text_surface, (x- text_surface.get_width()//4, y))
    pygame.display.update()
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

    # Create a drawing flag for dragging
    drawing = False
    testing = False

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle drawing on the grid
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left-click to draw
                    drawing = True
                elif event.button == 3:  # Right-click to clear the screen
                    screen.fill(WHITE)
                    testing = False

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False

            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    testing = True
                    x, y = pygame.mouse.get_pos()
                    grid_x = x // block_size
                    grid_y = y // block_size
                    pygame.draw.rect(screen, (0, 0, 0), (grid_x * block_size, grid_y * block_size, block_size, block_size))

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # When spacebar is pressed, get the grid array and print it
                    screen.fill(WHITE)
                    testing = False
        grid_array = get_grid_array(screen, grid_size, block_size)
        flat_array = concatenate_arrays(grid_array)
        output = nn.feedforward(flat_array)
        if testing:
            display_text(screen,lable[np.where(output == max(output))[0][0]] + " " + str((max(output)*100)//1), 50, width//2,20)
        else:
            display_text(screen, "Draw", 50,
                         width // 2, 20)
        pygame.display.flip()

    pygame.quit()



if __name__ == "__main__":
    learning_rate = .02
    train_data, test_data = Data.load_data(1000)
    T = Train(nn, learning_rate)
    T.train(train_data,test_data,5)
    main()





