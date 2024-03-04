#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:21:13 2024

@author: turbid
"""

import pygame
import random
import math
import matplotlib.pyplot as plt
from pygame.locals import DOUBLEBUF, OPENGL

class Triangle:
    def __init__(self, height, width):
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.points = [(random.randint(100//2, 700//2), random.randint(100//2, 500//2)) for _ in range(3)]
        self.velocity = [random.uniform(-2, 2), random.uniform(-2, 2)]
        self.height = height
        self.width = width
        self.angle = 0
        self.rotation_speed = random.uniform(-1e-1, 1e-1)

    def move(self):
        self.angle = self.rotation_speed
        for i in range(3):
            x, y = self.points[i]
            x += self.velocity[0]
            y += self.velocity[1]
            self.points[i] = (x, y)

            # Bounce off walls
            if x <= 0 or x >= self.width:
                self.velocity[0] *= -1
            if y <= 0 or y >= self.height:
                self.velocity[1] *= -1

    def rotate(self):
        center = sum(x for x, _ in self.points) / 3, sum(y for _, y in self.points) / 3
        sin_a = math.sin(math.radians(self.angle))
        cos_a = math.cos(math.radians(self.angle))

        for i in range(3):
            x, y = self.points[i]
            x -= center[0]
            y -= center[1]
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            self.points[i] = (new_x + center[0], new_y + center[1])

    def draw(self, surface):
        pygame.draw.polygon(surface, self.color, self.points)

class TriangleRenderer():
    def __init__(self):
        
        # Initialize Pygame
        pygame.init()
        
        # Window size
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        # pygame.display.set_caption("Moving Triangles")        
        
        clock = pygame.time.Clock()
        self.triangles = [Triangle(self.height, self.width) for _ in range(3)]
        
    def render(self):
        #%%#
        self.screen.fill((0, 0, 0))  # Clear screen
        
        for triangle in self.triangles:
            triangle.move()
            triangle.rotate()
            triangle.draw(self.screen)
        
        screen_array = pygame.surfarray.array3d(self.screen)
        pygame.display.flip()
        return screen_array

if __name__ == '__main__':
    triangler = TriangleRenderer()
    while True:
        triangler.render()





