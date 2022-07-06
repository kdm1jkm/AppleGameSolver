import game
from program import Program


def main():
    with Program() as p:
        p.turbo_mode = True
        p.start()


def test_game():
    game.test_board()
    game.test_play()


if __name__ == "__main__":
    main()
