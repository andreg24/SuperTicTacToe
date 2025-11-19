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

def render_tui(board):
  if len(board) != 81:
    raise ValueError("Board must be a list of 81 elements.")

  symbols = {0: " ", 1: "X", 2: "O"}

  h_border = "+" + "---+" * 9
  double_h_border = "+" + "━━━+" * 9
  v_border = "|"
  double_v_border = "┃"

  header = "  "
  for col in range(9):
    header += f"{col}   "
  print(header)

  for row in range(9):
      print(h_border if row % 3 != 0 else double_h_border)
      row_str = double_v_border
      for col in range(9):
          value = board[row * 9 + col]
          row_str += f" {symbols.get(value, ' ')} "
          row_str += v_border if (col + 1) % 3 != 0 else double_v_border
      row_str += f" {(row + 1) * 9}"
      print(row_str)
  print(double_h_border)