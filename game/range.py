from typing import Generator

from game.pos import Pos


class Range:
    def __init__(self, pos1: Pos, pos2: Pos):
        self.pos1 = Pos(min(pos1.x, pos2.x), min(pos1.y, pos2.y))
        self.pos2 = Pos(max(pos1.x, pos2.x), max(pos1.y, pos2.y))

    @property
    def width(self):
        return self.pos2.x - self.pos1.x + 1

    @property
    def height(self):
        return self.pos2.y - self.pos1.y + 1

    @property
    def size(self):
        return self.width * self.height

    def all_pos(self) -> Generator[Pos, None, None]:
        for y in range(self.pos1.y, self.pos2.y + 1):
            for x in range(self.pos1.x, self.pos2.x + 1):
                yield Pos(x, y)

    def __str__(self):
        return f"{self.pos1} to {self.pos2}"


def get_all_possible_range(_range: Range) -> Generator[Range, None, None]:
    for size in sorted(_range.all_pos(), key=lambda pos: (pos.x + 1) * (pos.y + 1)):
        end_pos = Pos(_range.width - size.x - 1, _range.height - size.y - 1)
        init_poses = Range(Pos(0, 0), end_pos).all_pos()
        for init_pos in init_poses:
            yield Range(init_pos, Pos(init_pos.x + size.x, init_pos.y + size.y))
