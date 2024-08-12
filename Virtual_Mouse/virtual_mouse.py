import cv2
import numpy as np
from HandsGestureDetector import HandDetector as hd
import time
import pyautogui
import math


class VirtualMouse:
    def __init__(self, wCam=1280, hCam=720, smoothing=10):
        self.wCam = wCam
        self.hCam = hCam
        self.smoothing = smoothing
        self.prevX = 0
        self.prevY = 0
        self.curX = 0
        self.curY = 0
        self.cap = self.initialize_camera()
        self.detector = hd(maxHands=1)
        self.wScr, self.hScr = pyautogui.size()
        self.mode = 'normal'  # Can be 'normal' or 'finger'
        self.cTime = 0
        self.pTime = 0
        self.over = False

    def initialize_camera(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, self.wCam)
        cap.set(4, self.hCam)
        return cap

    def set_mode(self, mode):
        if mode in ['normal', 'finger']:
            self.mode = mode

    def process_frame(self, img):
        img = cv2.flip(img, 1)
        hands, img = self.detector.findHands(img)
        self.lmList = self.detector.findPosition(img)

        if len(self.lmList):
            x1, y1 = self.lmList[8][1], self.lmList[8][2]
            x2, y2 = self.lmList[12][1], self.lmList[12][2]

            self.fingers = self.detector.fingersUp(hands[0])
            cv2.rectangle(img, (100, 100), (self.wCam - 50, self.hCam - 50), (255, 255, 0), 3)

            if self.mode == 'normal':
                if self.fingers[1] and not self.fingers[2]:
                    self.move_mouse(x1, y1, img)

                if self.fingers[1] and self.fingers[2]:
                    self.click_mouse(x1, y1, x2, y2, img)
            elif self.mode == 'finger':
                if self.fingers[1] and not self.fingers[2]:
                    self.move_mouse(x1, y1, img, finger_only=True)

                if self.fingers[1] and self.fingers[2]:
                    self.click_mouse(x1, y1, x2, y2, img)

                if self.fingers == [1, 1, 1, 1, 1] or self.fingers == [0, 1, 1, 1, 1]:
                    self.over = True

        self.display_fps(img)
        return img

    def move_mouse(self, x1, y1, img, finger_only=False):
        x3, y3 = np.interp(x1, (100, self.wCam - 100), (0, self.wScr)), np.interp(y1, (100, self.hCam - 100), (0, self.hScr))
        if finger_only:
            # self.curX = self.prevX + (x3 - self.prevX) // self.smoothing
            # self.curY = self.prevY + (y3 - self.prevY) // self.smoothing
            # pyautogui.moveTo(self.curX, self.curY)
            # self.prevX, self.prevY = self.curX, self.curY
            x1, y1 = self.lmList[8][1], self.lmList[8][2]
            # x1 = int(np.interp(x1, [1280 // 2, w], [0, 1280]))
            # y1 = int(np.interp(y1, [150, 720 - 150], [0, 720]))

            if self.fingers[1] and self.fingers[2]:
                cv2.circle(img, (x1, y1), 20, (0, 0, 255), -1)
        else:
            pyautogui.moveTo(x3, y3)
        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)

    def click_mouse(self, x1, y1, x2, y2, img):
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 60:
            pyautogui.click()

    def display_fps(self, img):
        self.cTime = time.time()
        fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    def run(self, cam=None):
        if cam:
            self.cap = cam
        while True:
            success, img = self.cap.read()
            if not success:
                break

            # cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), -1)

            img = self.process_frame(img)
            cv2.imshow('Virtual Mouse', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or self.over:
                self.over = False

                break
            elif key == ord('n'):
                self.set_mode('normal')
            elif key == ord('f'):
                self.set_mode('finger')
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vm = VirtualMouse()
    vm.run()
