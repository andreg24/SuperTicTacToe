"""
Module docstring
"""


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
