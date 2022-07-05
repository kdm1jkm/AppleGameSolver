import game
import mss
import cv2
import numpy as np


def resize_image(image: cv2.Mat, scale: int):
    return cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale))


def load_numbers(resize_scale: int, grayscale: bool):
    return [
        resize_image(
            cv2.cvtColor(
                cv2.imread(
                    f"resources/{i}.png",
                    cv2.IMREAD_UNCHANGED,
                ),
                cv2.COLOR_RGBA2GRAY if grayscale else cv2.COLOR_RGBA2RGB,
            ),
            resize_scale,
        )
        for i in range(1, 10)
    ]


def run(
    threshold: float = 0.994,
    resize_scale: int = 4,
    fps: int = 10,
    grayscale: bool = True,
    window_title: str = "AppleGame",
):
    screen_magnification: float = 1
    window_magnification: float = 0.5

    screen_mag_interval = 0.1
    window_mag_interval = 0.05

    with mss.mss() as sct:
        monitor_count = len(sct.monitors)
        monitor_index = 0

        monitor = sct.monitors[monitor_index]
        monitor["width"] = int(monitor["width"] * screen_magnification)
        monitor["height"] = int(monitor["height"] * screen_magnification)

        number_images = load_numbers(resize_scale, grayscale)
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_title,
            int(monitor["width"] * window_magnification),
            int(monitor["height"] * window_magnification),
        )

        while True:
            monitor = {
                "top": sct.monitors[monitor_index]["top"],
                "left": sct.monitors[monitor_index]["left"],
                "width": int(
                    sct.monitors[monitor_index]["width"] * screen_magnification
                ),
                "height": int(
                    sct.monitors[monitor_index]["height"] * screen_magnification
                ),
            }

            cv2.resizeWindow(
                window_title,
                int(monitor["width"] * window_magnification),
                int(monitor["height"] * window_magnification),
            )

            try:
                screen_image = resize_image(
                    cv2.cvtColor(
                        np.array(sct.grab(monitor)),
                        cv2.COLOR_RGBA2GRAY if grayscale else cv2.COLOR_RGBA2RGB,
                    ),
                    resize_scale,
                )
            except mss.ScreenShotError:
                screen_magnification = 1
                continue

            results: list[tuple[int, tuple[int, int]]] = []

            for i, number_image in enumerate(number_images):
                w = number_image.shape[1]
                h = number_image.shape[0]

                mt_result = cv2.matchTemplate(
                    screen_image, number_image, cv2.TM_CCORR_NORMED
                )

                for point in zip(*np.where(mt_result >= threshold)):
                    results.append(
                        (
                            i + 1,
                            (
                                monitor["left"]
                                + (point[1] + w // 2)
                                * resize_scale
                                // screen_magnification,
                                monitor["top"]
                                + (point[0] + h // 2)
                                * resize_scale
                                // screen_magnification,
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

            cv2.imshow(window_title, screen_image)

            results.sort(key=lambda result: result[1][1])
            results.sort(key=lambda result: result[1][0])

            # results를 보드로 변환하는 코드

            key = cv2.waitKey(1000 // fps)
            if key == ord("q"):
                cv2.destroyAllWindows()
                return

            elif key == ord("e"):
                grayscale = not grayscale
                number_images = load_numbers(resize_scale, grayscale)

            elif key == ord("a"):
                if monitor_index - 1 in range(monitor_count):
                    monitor_index -= 1
            elif key == ord("d"):
                if monitor_index + 1 in range(monitor_count):
                    monitor_index += 1

            elif key == ord("w"):
                screen_magnification += screen_mag_interval
            elif key == ord("s"):
                screen_magnification = max(
                    screen_mag_interval, screen_magnification - screen_mag_interval
                )

            elif key == ord("r"):
                window_magnification += window_mag_interval
            elif key == ord("f"):
                window_magnification = max(
                    window_mag_interval, window_magnification - window_mag_interval
                )

            elif key == ord("t"):
                resize_scale *= 2
                number_images = load_numbers(resize_scale, grayscale)
            elif key == ord("g"):
                resize_scale = max(1, resize_scale // 2)
                number_images = load_numbers(resize_scale, grayscale)


def main():
    run()


def test_game():
    game.test_board()
    game.test_play()


if __name__ == "__main__":
    main()
