
from nn_2 import NeuralNetwork as N
import random
from tqdm import tqdm
import pygame
import sys

pygame.init()

nn = N(2, [20,10,10], 1)
# training data
data = [
    ([0, 0],[0]),
    ([1, 1],[0]),
    ([1, 0],[1]),
    ([0, 1],[1])]

data2 = [
    ([0, .75],[0]),
    ([.8, 1],[1]),]

iterations = 50_000

def gtest_nn(data,iterations):
    for _ in tqdm(range(iterations)):
        inputs, target = random.choice(data)
        nn.train(inputs, target)

    # Test the trained neural network
    for i in range(len(data)):
        output = nn.feedforward(data[i][0])
        print(f"Input: {data[i][0]}, Output: {output}")


def display(data):
    WINDOW_WIDTH = 600
    WINDOW_HEIGHT = 600

    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("XOR")

    WHITE = (255, 255, 255)
    res = 5

    while True:
        iterations = 1000

        for _ in range(iterations):
            inputs, target = random.choice(data)
            nn.train(inputs, target)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("-----------------------------------------------------")

                    for i in range(len(data)):
                        output = nn.feedforward(data[i][0])
                        print(f"Input: {data[i][0]}, Output: {output}")




        window.fill(WHITE)

        for i in range(0, WINDOW_WIDTH, res):
            for j in range(0, WINDOW_HEIGHT, res):
                test = [i / WINDOW_WIDTH, j / WINDOW_HEIGHT]
                pygame.draw.rect(window, (
                int(255 * nn.feedforward(test)), int(255 * nn.feedforward(test)), int(255 * nn.feedforward(test))),
                                 (i, j, res, res))

        pygame.display.update()





if __name__ == '__main__':
    #test_nn(data,1000)
    nn= display(data)















