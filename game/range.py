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

    def all_pos(self) -> list[Pos]:
        return [
            Pos(x, y)
            for y in range(self.pos1.y, self.pos2.y + 1)
            for x in range(self.pos1.x, self.pos2.x + 1)
        ]

    def __str__(self):
        return f"{self.pos1} to {self.pos2}"


def get_all_possible_range(_range: Range) -> list[Range]:
    init_poses = _range.all_pos()
    range_end = Pos(_range.width - 1, _range.height - 1)
    return [
        Range(init_pos, end_pos)
        for init_pos in init_poses
        for end_pos in Range(init_pos, range_end).all_pos()
    ]
