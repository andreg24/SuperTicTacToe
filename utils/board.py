"""
DOCSTRING TO DO 
"""
from __future__ import annotations

from enum import Enum
import math
from abc import ABC, abstractmethod

import numpy as np
from .board_utils import absolute_to_relative, relative_to_absolute

class Status(Enum):
    PLAYER1_WIN = 1
    PLAYER2_WIN = 2
    TIE = -1
    GAME_NOT_OVER = 0

WINNING_COMBINATIONS = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]

# TRANSFORMATIONS
class BoardTransformation(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def grid_transform(self, grid, n, **kwargs):
        pass

    @abstractmethod
    def pos_transform(self, pos, n, **kwargs):
        pass

    def board_transform(self, super_cells, cells, pos, **kwargs):
        return self.grid_transform(super_cells, 3, **kwargs), self.grid_transform(cells, 9, **kwargs), self.pos_transform(pos, 3, **kwargs)

class BoardRotation(BoardTransformation):

    def __init__(self, angle):
        assert angle in [90, 180, 270], "angle not valid"
        self.angle = angle

    def grid_transform(self, grid, n):
        if n is None:
            n = int(math.isqrt(len(grid)))
        N = n*n

        if self.angle == 0:
            return grid[:]
        
        rotated_grid = [None] * N

        for idx, val in enumerate(grid):
            i, j = divmod(idx, n)  # row, col
            if self.angle == 90:
                new_idx = j * n + (n - 1 - i)
            elif self.angle == 180:
                new_idx = (n - 1 - i) * n + (n - 1 - j)
            elif self.angle == 270:
                new_idx = (n - 1 - j) * n + i
            rotated_grid[new_idx] = val

        return rotated_grid

    def pos_transform(self, pos, n):
        if pos == -1:
            return -1

        if self.angle == 0:
            return pos
        i, j = divmod(pos, n)  # row, col
        if self.angle == 90:
            return j * n + (n - 1 - i)
        elif self.angle == 180:
            return (n - 1 - i) * n + (n - 1 - j)
        else:
            return (n - 1 - j) * n + i
    
    def board_transform(self, super_cells, cells, pos):
        return self.grid_transform(super_cells, 3), self.grid_transform(cells, 9), self.pos_transform(pos, 3)

class BoardReflection(BoardTransformation):

    def __init__(self, is_horizontal):
        self.is_horizontal = is_horizontal

    def grid_transform(self, grid, n):
        if n is None:
            n = int(math.isqrt(len(grid)))
        N = n*n
        
        reflected_grid = [None] * N

        for idx, val in enumerate(grid):
            i, j = divmod(idx, n)  # row, col
            if self.is_horizontal:
                new_idx = i * n + (n - 1 - j)
            else:
                new_idx = (n - 1 - i) * n + j
            reflected_grid[new_idx] = val

        return reflected_grid

    def pos_transform(self, pos, n):
        if pos == -1:
            return -1

        i, j = divmod(pos, n)  # row, col
        if self.is_horizontal:
            return i * n + (n - 1 - j)
        else:
            return (n - 1 - i) * n + j

    def board_transform(self, super_cells, cells, pos):
        return self.grid_transform(super_cells, 3), self.grid_transform(cells, 9), self.pos_transform(pos, 3)

def rotate_grid(grid, k=1, n=None):
    """Rotate clockwise"""
    if n is None:
        N = len(grid)
        n = int(math.isqrt(N))
    else:
        N = n*n
    k %= 4
    if k == 0:
        return grid[:] # no rotatio
    
    rotated = [None] * N

    for idx, val in enumerate(grid):
        i, j = divmod(idx, n)  # row, col
        if k == 1:   # 90° CW
            new_idx = j * n + (n - 1 - i)
        elif k == 2: # 180°
            new_idx = (n - 1 - i) * n + (n - 1 - j)
        elif k == 3: # 270° CW
            new_idx = (n - 1 - j) * n + i
        rotated[new_idx] = val

    return rotated

def rotate_pos_cw(pos, grid, k=1, n=None):
    """Rotate clockwise"""
    if n is None:
        n = int(math.isqrt(len(grid)))

    k %= 4
    if k == 0:
        return pos
    
    rotated_pos = None

    i, j = divmod(pos, n)  # row, col
    if k == 1:   # 90° CW
        return j * n + (n - 1 - i)
    elif k == 2: # 180°
        return (n - 1 - i) * n + (n - 1 - j)
    else: # 270° CW
        return (n - 1 - j) * n + i

def reflect_grid(grid, is_horizontal, n=None):
    """k == 1 horizontal, k == -1 vertical"""
    if n is None:
        N = len(grid)
        n = int(math.isqrt(N))
    else:
        N = n*n
    
    reflected = [None] * N

    for idx, val in enumerate(grid):
        i, j = divmod(idx, n)  # row, col
        if is_horizontal:
            new_idx = i * n + (n - 1 - j)
        else:
            new_idx = (n - 1 - i) * n + j
        reflected[new_idx] = val

    return reflected

def reflect_pos(pos, grid, is_horizontal, n=None):
    """k == 1 horizontal, k == -1 vertical"""
    if n is None:
        n = int(math.isqrt(len(grid)))

    i, j = divmod(pos, n)  # row, col
    if is_horizontal:
        return i * n + (n - 1 - j)
    else:
        return (n - 1 - i) * n + j

TRANSFORMATIONS = [
    BoardRotation(90),
    BoardRotation(180),
    BoardRotation(270),
    BoardReflection(True),
    BoardReflection(False),
]

class SubTicTacToeBoard:

    def __init__(self):
        # self.squares holds a flat representation of the tic tac toe board.
        # an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0].
        # player 1's squares are marked 1, while player 2's are marked 2.
        # mapping of the flat indices to the 3x3 grid is as follows:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.cells = [0] * 9

    def copy(self):
      b = SubTicTacToeBoard()
      b.cells = [c for c in self.cells]
      return b

    @property
    def _n_empty_cells(self):
        """The current number of empty squares on the board."""
        return self.cells.count(0)

    def reset(self):
        """Remove all marks from the board."""
        self.cells = [0] * 9

    def play_turn(self, player, pos):
        """Place a mark by the player in the spot given.

        The following are required for a move to be valid:
        * The player must be a known player ID (either 0 or 1).
        * The spot must be be empty.
        * The spot must be in the board (integer: 0 <= spot <= 8)

        If any of those are not true, an assertion will fail.
        """
        assert 0 <= pos <= 8, "Invalid move location"
        assert player in [0, 1], "Invalid player"
        assert self.cells[pos] == 0, "Location is not empty"

        # player is [0, 1]. board values are stored as [1, 2].
        self.cells[pos] = player + 1

    def game_status(self):
        """Return status"""
        for indices in WINNING_COMBINATIONS:
            states = [self.cells[idx] for idx in indices]
            if states == [1, 1, 1]:
                return Status.PLAYER1_WIN
            if states == [2, 2, 2]:
                return Status.PLAYER2_WIN
        if self._n_empty_cells == 0:
            return Status.TIE
        return Status.GAME_NOT_OVER

    def __str__(self):
        return str(np.array(self.cells).reshape((3, 3)))

    def __repr__(self):
        return str(self)

    def legal_moves(self):
        """Return list of legal moves (as flat indices for spaces on the board)."""
        return [i for i, mark in enumerate(self.cells) if mark == 0]

class UltimateTicTacToeBoard:

    def __init__(self):
        # self.squares holds a flat representation of the super tic tac toe board
        # an empty board is [0, 0, 0, 0, 0, 0, 0, 0, 0].
        # player 1's victories are marked 1, while player 2's victories are marked 2
        # ties are marked -1
        # mapping of the flat indices to the 3x3 grid is as follows:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        # COULD START USING CELL AND GRID NOMENCLATURE INSTEAD
        self.super_cells = [0] * 9 # high level board
        self.cells = [0] * 81 # board accounting for all the possible squares
        self.sub_boards = [SubTicTacToeBoard() for _ in range(9)]
        self.current_pos = -1 # -1 indicates free choice
        self.current_player = 0

    def copy(self):
      b = UltimateTicTacToeBoard()
      b.super_cells = [sc for sc in self.super_cells]
      b.cells = [c for c in self.cells]
      b.sub_boards = [c.copy() for c in self.sub_boards]
      b.current_pos = self.current_pos
      b.current_player = self.current_player
      return b

    @property
    def _n_empty_super_cells(self):
        """The current number of empty squares on the board."""
        return self.super_cells.count(0)

    def reset(self):
        """Remove all marks from the board."""
        self.super_cells = [0] * 9
        self.cells = [0] * 81
        for sub_board in self.sub_boards:
            sub_board.reset()
        self.current_pos = -1 # -1 indicates free choice
        self.current_player = 0

    def play_turn(self, player: int, pos: int):
        """Place a mark by the player in the spot given.

        The following are required for a move to be valid:
        * The player must be a known player ID (either 0 or 1).
        * The spot must be be empty.
        * The spot must be in the board (integer: 0 <= spot <= 8)

        If any of those are not true, an assertion will fail.
        """
        assert player == self.current_player
        assert 0 <= pos <= 80
        assert self.is_valid_move(player, pos), f"player={player}, current={self.current_player}, pos={pos}"


        # player is [0, 1]. board values are stored as [1, 2].
        self.cells[pos] = player + 1
        super_pos, sub_pos = absolute_to_relative(pos)
        sub_board = self.sub_boards[super_pos]
        sub_board.play_turn(player, sub_pos)
        sub_board_status = sub_board.game_status()
        self.super_cells[super_pos] = sub_board_status.value
        # update current pos
        self.current_pos = sub_pos if self.super_cells[sub_pos] == 0 else -1
        self.current_player = (self.current_player + 1) % 2 # 0 -> 1, 1->0

    def is_valid_move(self, player: int, pos: int):
        # correct player
        if player != self.current_player:
            return False
        # valid pos range
        if not (0 <= pos <= 80):
            return False
        super_pos, sub_pos = absolute_to_relative(pos)
        # wrong uper pos
        if self.current_pos != -1 and self.current_pos != super_pos:
            return False
        # super pos already occupied
        if self.super_cells[super_pos] != 0:
            return False
        # sub pos occupied
        if self.sub_boards[super_pos].cells[sub_pos] != 0:
            return False
        return True

    def game_status(self):
        """Return status (winner, TTT_TIE if no winner, or TTT_GAME_NOT_OVER)."""
        for indices in WINNING_COMBINATIONS:
            states = [self.super_cells[idx] for idx in indices]
            if states == [1, 1, 1]:
                return Status.PLAYER1_WIN
            if states == [2, 2, 2]:
                return Status.PLAYER2_WIN
        if self._n_empty_super_cells == 0:
            return Status.TIE
        return Status.GAME_NOT_OVER

    def __str__(self):
        return str(np.array(self.super_cells).reshape((3, 3)))

    def __repr__(self):
        return str(self)

    def legal_moves(self):
        """Return list of legal moves (as flat indices for spaces on the board)."""
        available_moves = []
        if self.current_pos == -1:
            for super_pos in range(9):
                if self.super_cells[super_pos] == 0:
                    sub_board = self.sub_boards[super_pos]
                    for sub_pos in sub_board.legal_moves():
                        pos = relative_to_absolute(super_pos, sub_pos)
                        available_moves.append(pos)
        else:
            super_pos = self.current_pos
            sub_board = self.sub_boards[super_pos]
            for sub_pos in sub_board.legal_moves():
                pos = relative_to_absolute(super_pos, sub_pos)
                available_moves.append(pos)
        return available_moves

    def apply_transformation(self, transformation):
        self.super_cells, self.cells, self.current_pos = transformation.board_transform(self.super_cells, self.cells, self.current_pos)
        self._propagate_to_sub_boards()

    def _propagate_to_sub_boards(self):
        cells_np = np.array(self.cells).reshape(9, 9)
        for p in range(9):
            i, j = divmod(p, 3)
            self.sub_boards[p].cells = list(cells_np[i*3:(i+1)*3, j*3:(j+1)*3].flatten())