from random import randint

import cv2
import mss
import numpy as np
from cv2 import Mat
import pyautogui as pg

from game import Board
from game.pos import Pos


class Program:
    def __init__(self) -> None:
        self.threshold: float = 0.994
        self.fps: int = 10
        self.resize_scale: int = 4
        self.grayscale: bool = True
        self.window_title: str = "AppleGame"
        self.screen_magnification: float = 1
        self.window_magnification: float = 0.5
        self.screen_mag_interval: float = 1
        self.window_mag_interval: float = 0.05
        self.load_numbers()
        pg.FAILSAFE = False
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)

    def __enter__(self):
        self.sct = mss.mss()
        self.monitor_index = 0
        self.monitor_count = len(self.sct.monitors)
        self.config_monitor()
        return self

    def __exit__(self, *_):
        self.sct.__exit__()

    def load_numbers(self):
        self.number_images = [
            resize_image(
                cv2.cvtColor(
                    cv2.imread(
                        f"resources/{i}.png",
                        cv2.IMREAD_UNCHANGED,
                    ),
                    cv2.COLOR_RGBA2GRAY if self.grayscale else cv2.COLOR_RGBA2RGB,
                ),
                self.resize_scale,
            )
            for i in range(1, 10)
        ]

    def start(self):
        self.load_numbers()
        self.config_opencv_window()
        self.config_monitor()

        while True:
            try:
                screen_image = self.get_screen_image()
            except mss.ScreenShotError:
                self.screen_magnification = 1
                continue

            results = self.extract_positions(screen_image)

            cv2.imshow(self.window_title, screen_image)

            board = self.convert_board(results)
            print(board)
            print("-" * 50)

            key = cv2.waitKey(1000 // self.fps)
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            else:
                self.handle_control(key)

    def convert_board(
        self,
        results: list[tuple[int, tuple[int, int]]],
        width: int = 18,
        height: int = 9,
    ) -> Board:
        if len(results) == 0:
            return Board(width, height)
        left, right, top, bottom = self.calc_end_position(results)

        game_width = right - left
        game_height = bottom - top
        horizontal_interval: float = game_width / (width - 1)
        vertical_interval: float = game_height / (height - 1)

        converted_results: dict[Pos, int] = {}
        for result in results:
            x = round((result[1][0] - left) / horizontal_interval)
            y = round((result[1][1] - top) / vertical_interval)
            converted_results[Pos(x, y)] = result[0]

        board = Board(width, height)

        def value_from_index(index: int):
            pos = board.index_to_pos(index)
            if pos in converted_results.keys():
                return converted_results[pos]
            else:
                return 0

        board.fill(value_from_index)
        return board

    def calc_end_position(self, results):
        left = min(results, key=lambda result: result[1][0])[1][0]
        right = max(results, key=lambda result: result[1][0])[1][0]
        top = min(results, key=lambda result: result[1][1])[1][1]
        bottom = max(results, key=lambda result: result[1][1])[1][1]
        return left, right, top, bottom

    def handle_control(self, key):
        if key == ord("e"):
            grayscale = not grayscale
            self.load_numbers()

        elif key == ord("a"):
            if self.monitor_index - 1 in range(self.monitor_count):
                self.monitor_index -= 1
                self.config_monitor()
        elif key == ord("d"):
            if self.monitor_index + 1 in range(self.monitor_count):
                self.monitor_index += 1
                self.config_monitor()

        elif key == ord("w"):
            self.screen_magnification += self.screen_mag_interval
            self.config_monitor()
        elif key == ord("s"):
            self.screen_magnification = max(
                self.screen_mag_interval,
                self.screen_magnification - self.screen_mag_interval,
            )
            self.config_monitor()

        elif key == ord("r"):
            self.window_magnification += self.window_mag_interval
            self.config_opencv_window()
        elif key == ord("f"):
            self.window_magnification = max(
                self.window_mag_interval,
                self.window_magnification - self.window_mag_interval,
            )
            self.config_opencv_window()

        elif key == ord("t"):
            self.resize_scale *= 2
            self.config_monitor()
            self.load_numbers()
        elif key == ord("g"):
            self.resize_scale = max(1, self.resize_scale // 2)
            self.config_monitor()
            self.load_numbers()

    def extract_positions(self, screen_image: Mat):
        results: list[tuple[int, tuple[int, int]]] = []
        for i, number_image in enumerate(self.number_images):
            w = number_image.shape[1]
            h = number_image.shape[0]

            mt_result = cv2.matchTemplate(
                screen_image, number_image, cv2.TM_CCORR_NORMED
            )

            for point in zip(*np.where(mt_result >= self.threshold)):
                results.append(
                    (
                        i + 1,
                        (
                            self.monitor["left"]
                            + (point[1] + w // 2)
                            * self.resize_scale
                            // self.screen_magnification,
                            self.monitor["top"]
                            + (point[0] + h // 2)
                            * self.resize_scale
                            // self.screen_magnification,
                        ),
                    )
                )
                cv2.rectangle(
                    screen_image,
                    point[::-1],
                    (point[1] + w, point[0] + h),
                    (0, 0, 255),
                    1,
                )

        results.sort(key=lambda result: result[1][1])
        results.sort(key=lambda result: result[1][0])
        return results

    def get_screen_image(self):
        return resize_image(
            cv2.cvtColor(
                np.array(self.sct.grab(self.monitor)),
                cv2.COLOR_RGBA2GRAY if self.grayscale else cv2.COLOR_RGBA2RGB,
            ),
            self.resize_scale,
        )

    def config_opencv_window(self):
        cv2.resizeWindow(
            self.window_title,
            int(self.monitor["width"] * self.window_magnification),
            int(self.monitor["height"] * self.window_magnification),
        )

    def config_monitor(self):
        self.monitor = {
            "top": self.sct.monitors[self.monitor_index]["top"],
            "left": self.sct.monitors[self.monitor_index]["left"],
            "width": int(
                self.sct.monitors[self.monitor_index]["width"]
                * self.screen_magnification
            ),
            "height": int(
                self.sct.monitors[self.monitor_index]["height"]
                * self.screen_magnification
            ),
        }
        self.config_opencv_window()


def resize_image(image: cv2.Mat, scale: int):
    return cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale))
