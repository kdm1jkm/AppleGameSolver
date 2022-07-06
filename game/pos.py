class Pos:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, __o: object) -> bool:
        if type(__o) is type(self):
            return self.x == __o.x and self.y == __o.y
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
