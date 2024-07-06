import cv2
from cvzone.HandTrackingModule import HandDetector as hd
import os
import numpy as np

class VirtualPainter:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)
        self.detector = hd()

        self.folder = r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Images'
        print(os.getcwd())
        # print(os.listdir(self.folder))
        self.header_images = [cv2.imread(f'{self.folder}/{img}') for img in os.listdir(self.folder)]
        self.header = self.header_images[0]

        self.xp, self.yp = 0, 0
        self.brush_thickness = 30
        self.eraser_thickness = 100
        self.color1 = (255, 192, 203)
        self.color2 = self.color3 = (0, 0, 0)
        self.selected = ''
        self.circle_flag = False
        self.done = False
        self.doneL = False
        self.gone = False
        self.line_flag = False
        self.lm_list = []

        self.circle_x1, self.circle_y1, self.radius = 0, 0, 0
        self.line_start, self.line_end = (0, 0), (0, 0)
        self.img_canvas = np.zeros((720, 1280, 3), np.uint8)

    def draw(self, cam):
        self.cam = cam
        while True:
            _, img = self.cam.read()
            if _:
                img = cv2.flip(img, 1)
                hands, img = self.detector.findHands(img, flipType=False)

                if hands:
                    self.lm_list = self.detector.findPosition(img)
                    if self.lm_list:
                        self.process_hand_gestures(img, hands)

                img_gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
                _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(img, img_inv)
                img = cv2.bitwise_or(img, self.img_canvas)

                img[:104, :1007] = self.header

                cv2.imshow('img', img)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            else:
                break
        cv2.destroyAllWindows()

    def process_hand_gestures(self, img, hands):
        x1, y1 = self.lm_list[8][1], self.lm_list[8][2]
        x2, y2 = self.lm_list[12][1:]
        fingers = self.detector.fingersUp(hands[0])

        if fingers[1] and fingers[2]:
            self.select_tool(x1, y1, x2, y2, img)

        elif fingers[1] and not fingers[2]:
            self.draw_on_canvas(img, hands)

    def select_tool(self, x1, y1, x2, y2, img):

        if y1 < 130:
            # self.header = self.header_images[0]
            if 10 < x1 < 100:
                self.header = self.header_images[0]
                self.color1 = (255, 192, 203)
                self.color3 = (255, 192, 203)
                self.selected = 'brush1'
            elif 200 < x1 < 300:
                self.header = cv2.resize(self.header_images[1], (1007, 104), interpolation=cv2.INTER_AREA)
                self.color1 = (0, 0, 255)
                self.color3 = (0, 0, 255)
                self.selected = 'brush2'
            elif 450 < x1 < 550:
                self.select_circle()
            elif 600 < x1 < 700:
                self.select_line()
            elif 800 < x1 < 900:
                self.color1 = (0, 0, 0)
                self.selected = 'eraser'

        cv2.line(img, (x1, y1), (x2, y2), self.color3, 3)

    def select_circle(self):
        self.color2 = (0, 0, 0)
        self.color1 = (0, 0, 0)
        self.color3 = (255, 0, 0)
        self.selected = 'circle'
        self.circle_flag = True
        self.done = False

    def select_line(self):
        self.color2 = (0, 0, 0)
        self.color1 = (0, 0, 0)
        self.color3 = (0, 255, 0)
        self.selected = 'line'
        self.line_flag = True
        self.doneL = False

    def draw_on_canvas(self, img, hands):
        x1, y1 = self.lm_list[8][1], self.lm_list[8][2]
        cv2.circle(img, (x1, y1), 10, (255, 255, 255), -1)
        if self.xp == 0 and self.yp == 0:
            self.xp, self.yp = x1, y1

        self.draw_line(x1, y1, img)
        if self.selected == 'eraser':
            self.draw_eraser(x1, y1, img)
        elif self.selected == 'circle':
            self.draw_circle(x1, y1, img, hands)
        elif self.selected == 'line':
            self.draw_line_shape(x1, y1, img, hands)
        self.xp, self.yp = x1, y1

    def draw_line(self, x1, y1, img):
        cv2.line(img, (self.xp, self.yp), (x1, y1), self.color1, self.brush_thickness)
        cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.color1, self.brush_thickness)
        self.xp, self.yp = x1, y1

    def draw_eraser(self, x1, y1, img):
        cv2.line(img, (self.xp, self.yp), (x1, y1), self.color1, self.eraser_thickness)
        cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.color1, self.eraser_thickness)
        self.xp, self.yp = x1, y1

    def draw_circle(self, x1, y1, img, hands):
        if len(hands) == 2 and self.circle_flag:
            self.circle_x1, self.circle_y1 = x1, y1
            lm_list2 = self.detector.findPosition(img, 1)
            thumbX, thumbY = lm_list2[8][1], lm_list2[8][2]
            self.radius = int(((thumbX - x1) ** 2 + (thumbY - y1) ** 2) ** 0.5)
            x3, y3 = self.lm_list[4][1], self.lm_list[4][2]
            length = int(((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5)
            if length < 160:
                self.circle_flag = False
                self.done = True
                self.color2 = (255, 0, 0)
                cv2.circle(img, (self.circle_x1, self.circle_y1), self.radius, self.color2, 5)
                cv2.circle(self.img_canvas, (self.circle_x1, self.circle_y1), self.radius, self.color2, 5)
        if not self.done:
            cv2.circle(img, (self.circle_x1, self.circle_y1), self.radius, self.color2, 5)
            cv2.circle(self.img_canvas, (self.circle_x1, self.circle_y1), self.radius, self.color2, 5)

    def draw_line_shape(self, x1, y1, img, hands):
        if len(hands) == 2 and self.line_flag:
            self.line_start = (x1, y1)
            lm_list2 = self.detector.findPosition(img, 1)
            self.line_end = (lm_list2[8][1], lm_list2[8][2])
            x3, y3 = self.lm_list[4][1], self.lm_list[4][2]
            length = int(((x3 - x1) ** 2 + (y3 - y1) ** 2) ** 0.5)

            if length < 160:
                self.line_flag = False
                self.doneL = True
                self.color2 = (255, 0, 0)
                cv2.line(img, self.line_start, self.line_end, self.color2, 5)
                cv2.line(self.img_canvas, self.line_start, self.line_end, self.color2, 5)
        if not self.doneL:
            cv2.line(img, self.line_start, self.line_end, self.color2, 5)
            cv2.line(self.img_canvas, self.line_start, self.line_end, self.color2, 5)


if __name__ == '__main__':
    painter = VirtualPainter()
    painter.draw()
