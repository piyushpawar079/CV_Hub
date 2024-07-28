import cv2
import numpy as np
import os
from cvzone.HandTrackingModule import HandDetector as hd

class PresentationController:
    def __init__(self, folder=r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Presentations'):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)
        self.detector = hd(maxHands=1)
        self.folder = folder
        self.images = os.listdir(folder)
        self.img_number = 0
        self.ws, self.hs = 213, 120
        self.threshold = 425
        self.buttonPressed = False
        self.buttonCounter = 0
        self.buttonDelay = 10
        self.annotations = [[]]
        self.annotationsNumber = -1
        self.annotationsFlag = False
        self.over = False

    def load_image(self):
        current_image = os.path.join(self.folder, self.images[self.img_number])
        img_current = cv2.imread(current_image)
        img_current = cv2.resize(img_current, (1280, 720), interpolation=cv2.INTER_AREA)
        return img_current

    def process_frame(self, img, img_current):
        hands, img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img)
        cv2.line(img, (0, self.threshold), (1280, self.threshold), (0, 255, 0), 10)

        if hands:
            fingers = self.detector.fingersUp(hands[0])

        if lmList and not self.buttonPressed:
            cx, cy = hands[0]['center']
            if cy <= self.threshold:
                if fingers[0]:
                    self.change_slide(-1)
                elif fingers[4]:
                    self.change_slide(1)

            x1, y1 = self.map_coordinates(lmList[8][1], lmList[8][2])
            if fingers[1] and fingers[2]:
                self.annotationsFlag = False
                cv2.circle(img_current, (x1, y1), 20, (0, 0, 255), -1)
            elif fingers[1] and not fingers[2]:
                self.draw_annotation(x1, y1, img_current)
            else:
                self.annotationsFlag = False
            if fingers == [0, 1, 1, 1, 0]:
                self.remove_last_annotation()

            if fingers == [1, 1, 1, 1, 1]:
                self.over = True

        if self.buttonPressed:
            self.buttonCounter += 1
            if self.buttonCounter > self.buttonDelay:
                self.buttonCounter = 0
                self.buttonPressed = False

        self.draw_annotations(img_current)
        return img, img_current

    def change_slide(self, direction):
        self.buttonPressed = True
        self.img_number = max(0, min(self.img_number + direction, len(self.images) - 1))
        self.annotations = [[]]
        self.annotationsNumber = -1
        self.annotationsFlag = False

    def map_coordinates(self, x1, y1):
        x1 = int(np.interp(x1, [1280 // 2, 1280], [0, 1280]))
        y1 = int(np.interp(y1, [150, 720 - 150], [0, 720]))
        return x1, y1

    def draw_annotation(self, x1, y1, img_current):
        if not self.annotationsFlag:
            self.annotationsFlag = True
            self.annotationsNumber += 1
            self.annotations.append([])
        cv2.circle(img_current, (x1, y1), 20, (0, 0, 255), -1)
        self.annotations[self.annotationsNumber].append([x1, y1])

    def remove_last_annotation(self):
        if self.annotations:
            self.annotations.pop(-1)
            self.annotationsNumber -= 1
            self.buttonPressed = True

    def draw_annotations(self, img_current):
        for i in range(len(self.annotations)):
            for j in range(len(self.annotations[i])):
                if j:
                    cv2.line(img_current, self.annotations[i][j - 1], self.annotations[i][j], (0, 0, 200), 10)

    def run(self, cam):
        self.cam = cam
        while True:
            _, img = self.cam.read()
            if not _:
                break
            img = cv2.flip(img, 1)
            img_current = self.load_image()
            img, img_current = self.process_frame(img, img_current)
            img_Small = cv2.resize(img, (self.ws, self.hs))
            img_current[:self.hs, 1280 - self.ws:1280] = img_Small
            cv2.imshow('Presentation', img_current)
            key = cv2.waitKey(1)
            if key == ord('q') or self.over:
                self.over = False

                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = PresentationController()
    controller.run()
