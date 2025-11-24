import os
import pygame


def get_image(path):
    """Return a pygame image loaded from the given path."""

    cwd = os.path.dirname(__file__)
    image = pygame.image.load(cwd + "/../" + path)
    return image


def get_font(path, size):
    """Return a pygame font loaded from the given path."""

    cwd = os.path.dirname(__file__)
    font = pygame.font.Font((cwd + "/../" + path), size)
    return font
