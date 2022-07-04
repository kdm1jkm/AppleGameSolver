from functools import reduce
from random import randint
from typing import Callable

from game.pos import Pos

from game.range import Range, get_all_possible_range


class Board:
    def __init__(self, width: int, height: int) -> None:
        self.__data: list[int] = []
        self.width = width
        self.height = height
        self.fill(lambda _: 0)

    def fill(self, func: Callable[[int], int]) -> None:
        self.__data = [func(i) for i in range(self.width * self.height)]

    def fill_range(self, _range: Range, func: Callable[[int], int]) -> None:
        for i, pos in enumerate(_range.all_pos()):
            self.set_data(pos, func(i))

    def sub_board(self, _range: Range):
        board = Board(_range.width, _range.height)
        board.__data = [self.__data[self.pos_to_index(pos)] for pos in _range.all_pos()]
        return board

    def find_all_target_sum_range(self, target: int = 10) -> list[Range]:
        ranges = get_all_possible_range(self.get_range())
        valid_ranges = filter(
            lambda _range: self.calc_range_sum(_range) == target, ranges
        )
        return list(valid_ranges)

    def find_one_target_sum_range(self, target: int = 10) -> Range:
        ranges = get_all_possible_range(self.get_range())
        for _range in ranges:
            if self.calc_range_sum(_range) == target:
                return _range

        return None

    def calc_sum(self) -> int:
        return sum(self.__data)

    def calc_range_sum(self, _range: Range) -> int:
        return self.sub_board(_range).calc_sum()

    def index_to_pos(self, index: int) -> Pos:
        return Pos(index % self.width, index // self.width)

    def pos_to_index(self, pos: Pos) -> int:
        return pos.x + pos.y * self.width

    def get_data(self, pos: Pos) -> int:
        return self.__data[self.pos_to_index(pos)]

    def set_data(self, pos: Pos, new_data: int) -> None:
        self.__data[self.pos_to_index(pos)] = new_data

    def is_pos_valid(self, pos: Pos) -> bool:
        return pos.x in range(self.width) and pos.y in range(self.height)

    def get_range(self) -> Range:
        return Range(Pos(0, 0), Pos(self.width - 1, self.height - 1))

    def __str__(self):
        return reduce(
            lambda a, b: f"{a}\n{b}",
            map(
                lambda y: reduce(
                    lambda a, b: f"{a} {b}",
                    self.__data[
                        self.pos_to_index(Pos(0, y)) : self.pos_to_index(Pos(0, y + 1))
                    ],
                ),
                range(self.height),
            ),
        )


def test_board():
    board = Board(5, 5)
    board.fill(lambda _: randint(1, 9))
    print(board)
    print("-" * 50)

    r = Range(Pos(1, 1), Pos(3, 4))
    print(r)
    print(*r.all_pos(), sep=" ")
    sb = board.sub_board(r)
    print(sb)
    print("-" * 50)

    print(board.calc_range_sum(r))
    print(sb.calc_sum())
    print("-" * 50)

    print(board)
    print(*board.find_all_target_sum_range(), sep="\n")
    print("-" * 50)


def test_play():
    board = Board(10, 5)
    board.fill(lambda _: randint(1, 9))

    while True:
        print(board)
        target_range = board.find_one_target_sum_range()
        if target_range == None:
            break
        print(target_range)
        print(board.sub_board(target_range))
        board.fill_range(target_range, lambda _: 0)
        print("-" * 50)
