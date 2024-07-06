import cv2
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector as hd
from CV_Desktop.VolumeControl import VolumeControl
from CV_Desktop.Paint_app import VirtualPainter
from CV_Desktop.Presentation_app import PresentationController
from CV_Desktop.Pong_app import PongGame
from CV_Desktop.Virtual_Mouse import VirtualMouse
from CV_Desktop.Math_AI_app import HandGestureAI


class CVApp:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)

        self.detector = hd(maxHands=1)
        self.vc = VolumeControl()
        self.vp = VirtualPainter()
        self.pc = PresentationController()
        self.vm = VirtualMouse()
        self.pg = PongGame()
        api_key = 'AIzaSyBLoq2qPnvxqfGfyqZb2ifo202nkcPPKKA'
        self.m_ai = HandGestureAI(api_key)

        self.over = False

    def loading(self):
        start_time = time.time()
        load_duration = 3  # Duration in seconds
        window_created = False

        while time.time() - start_time < load_duration:
            ret, img = self.cam.read()
            if not ret:
                continue
            img = cv2.flip(img, 1)

            progress = int((time.time() - start_time) / load_duration * 800) + 150
            angle = (time.time() - start_time) * 2 * np.pi  # Full rotation every second

            # Draw loading bar
            cv2.rectangle(img, (150, 300), (950, 400), (0, 255, 0), 5)
            cv2.rectangle(img, (150, 300), (progress, 400), (0, 255, 0), -1)
            cv2.rectangle(img, (progress, 295), (progress + 10, 405), (0, 255, 255), -1)

            # Draw rotating circle
            center = (550, 250)
            radius = 30
            circle_x = int(center[0] + radius * np.cos(angle))
            circle_y = int(center[1] + radius * np.sin(angle))
            cv2.circle(img, center, radius, (0, 0, 255), 2)
            cv2.circle(img, (circle_x, circle_y), 10, (255, 0, 0), -1)

            # Display loading text
            cv2.putText(img, "Loading...", (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Loading', img)
            cv2.waitKey(1)

            # Set flag when window is created
            if not window_created:
                window_created = True

        # Destroy the 'Loading' window only if it was created
        if window_created:
            cv2.destroyWindow('Loading')

    def draw_interface(self, img):
        buttons = [(100, 100, 200, 200, 'Volume Control'),
                   (300, 100, 400, 200, 'Paint App'),
                   (500, 100, 600, 200, 'Math AI App'),
                   (700, 100, 800, 200, 'Presentation App'),
                   (100, 300, 200, 400, 'Pong Game'),
                   (300, 300, 400, 400, 'Virtual Mouse')]

        for (x1, y1, x2, y2, label) in buttons:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            lines = label.split()
            for i, line in enumerate(lines):
                cv2.putText(img, line, (x1 - 10, y1 + 150 + i * 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    def run(self):
        while True:
            ret, img = self.cam.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            hands, img = self.detector.findHands(img, flipType=False)

            if hands:
                lmlist = self.detector.findPosition(img)
                fingers = self.detector.fingersUp(hands[0])

                x1, y1 = lmlist[8][1], lmlist[8][2]
                x2, y2 = lmlist[12][1], lmlist[12][2]
                if fingers == [1, 1, 1, 1, 1]:
                    self.over = True

                if fingers[1] and fingers[2]:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 5)
                elif fingers[1] and not fingers[2]:
                    cv2.circle(img, (x1, y1), 20, (0, 0, 255), -1)
                    if 100 < x1 < 200 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.vc.run(self.cam)
                        self.loading()
                    elif 300 < x1 < 400 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.vp.draw(self.cam)
                        self.loading()
                    elif 700 < x1 < 800 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.pc.run(self.cam)
                        self.loading()
                    elif 100 < x1 < 200 and 300 < y1 < 400:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.pg.play_game(self.cam)
                        self.loading()
                    elif 300 < x1 < 400 and 300 < y1 < 400:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.vm.run(self.cam)
                        self.loading()
                    elif 500 < x1 < 600 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.m_ai.run_app(self.cam)
                        self.loading()

            self.draw_interface(img)
            cv2.imshow('img', img)
            cv2.waitKey(1)
            if self.over:
                break


if __name__ == "__main__":
    app = CVApp()
    app.run()
