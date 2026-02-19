"""
Module docstring
"""

import torch


def pos_to_coordinates(pos, size=3):
    """
    Convert a flat row-major index into (row, column) coordinates on a square grid of given size.
    """
    return pos // size, pos % size


def coordinates_to_pos(coordinates, size=3):
    """
    Convert (row, column) coordinates into flat row-major index on a square grid of given size.
    """
    return size * coordinates[0] + coordinates[1]


def absolute_to_relative(square_pos):
    """
    Convert an absolute index in a 9x9 grid into
    relative positions for the 3x3 sub-board and the cell within that sub-board.
    """
    square_row, square_col = pos_to_coordinates(square_pos, 9)
    super_row, super_col = square_row // 3, square_col // 3
    super_pos = coordinates_to_pos((super_row, super_col))

    sub_row, sub_col = square_row - 3 * super_row, square_col - 3 * super_col
    sub_pos = coordinates_to_pos((sub_row, sub_col))
    return super_pos, sub_pos


def relative_to_absolute(super_pos, sub_pos):
    """
    Convert relative positions for 3x3 sub-board and the cell within that sub-board into
    an absolute index in a 9x9
    """
    super_row, super_col = pos_to_coordinates(super_pos)
    sub_row, sub_col = pos_to_coordinates(sub_pos)
    square_row, square_col = sub_row + 3 * super_row, sub_col + 3 * super_col
    square_pos = coordinates_to_pos((square_row, square_col), 9)
    return square_pos

# torch batches

def split_subboards(t: torch.Tensor):
    """
    t: (B, C, 9, 9)
    returns: (B, 9, C, 3, 3)
    """
    B, C, _, _ = t.shape
    y = t.view(B, C, 3, 3, 3, 3)          # (B,C, sb_r, cell_r, sb_c, cell_c)
    y = y.permute(0, 2, 4, 1, 3, 5)       # (B, sb_r, sb_c, C, cell_r, cell_c)
    y = y.reshape(B, 9, C, 3, 3)          # (B, 9, C, 3, 3)
    return y

def merge_subboards(t):
    """
    t: (B, 9, C, 3, 3)
    returns: (B, C, 9, 9)
    """
    B, _, C, _, _ = t.shape
    y = t.view(B, 3, 3, C, 3, 3)         # (B, sb_r, sb_c, C, cell_r, cell_c)
    y = y.permute(0, 3, 1, 4, 2, 5)       # (B, C, sb_r, cell_r, sb_c, cell_c)
    y = y.reshape(B, C, 9, 9)
    return y

class Rotation:

    def __init__(self, k):
        self.k = k
    
    def transform_batch(self, t: torch.Tensor):
        """batch (..., 9, 9)"""
        return torch.rot90(t, self.k, dims=(-1, -2))

    def inverse_transform_batch(self, t: torch.Tensor):
        """batch (..., 9, 9)"""
        return torch.rot90(t, -self.k, dims=(-1, -2))

    def rotate_action_90(self, action: int, N: int = 9):
        r, c = divmod(action, N)
        r2, c2 = c, N - 1 - r   # clockwise
        return r2 * N + c2

    def transform_action(self, action):
        for i in range(self.k):
            action = self.rotate_action_90(action)
        return action

    def inverse_transform_action(self, action: int, N: int = 9):
        for _ in range((4 - self.k) % 4):
            action = self.rotate_action_90(action, N)
        return action

class Reflection:
    """
    Reflection symmetries for NxN board.

    mode:
        0 = identity
        1 = horizontal flip (left-right)
        2 = vertical flip (up-down)
        3 = main diagonal transpose
        4 = anti-diagonal reflection
    """

    def __init__(self, k: int, N: int = 9):
        assert k in [0, 1, 2, 3, 4]
        self.k = k
        self.N = N

    def transform_batch(self, t: torch.Tensor):
        """t shape (..., N, N)"""
        if self.k == 0:
            return t
        elif self.k == 1:  # left-right
            return torch.flip(t, dims=(-1,))
        elif self.k == 2:  # up-down
            return torch.flip(t, dims=(-2,))
        elif self.k == 3:  # main diagonal
            return t.transpose(-2, -1)
        elif self.k == 4:  # anti-diagonal
            return torch.flip(t.transpose(-2, -1), dims=(-1, -2))

    def inverse_transform_batch(self, t: torch.Tensor):
        """Reflections are self-inverse"""
        return self.transform_batch(t)

    def transform_action(self, action: int):
        """action is flattened index in [0, N*N)"""
        r, c = divmod(action, self.N)

        if self.k == 0:      # identity
            r2, c2 = r, c
        elif self.k == 1:    # left-right
            r2, c2 = r, self.N - 1 - c
        elif self.k == 2:    # up-down
            r2, c2 = self.N - 1 - r, c
        elif self.k == 3:    # main diagonal
            r2, c2 = c, r
        elif self.k == 4:    # anti-diagonal
            r2, c2 = self.N - 1 - c, self.N - 1 - r

        return r2 * self.N + c2

    def inverse_transform_action(self, action: int):
        """Reflections are self-inverse"""
        return self.transform_action(action)
