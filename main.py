import game
from screenreader.reader import test


def main():
    test_screenreader()


def test_screenreader():
    test()


def test_game():
    game.test_board()
    game.test_play()


if __name__ == "__main__":
    main()
