import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import random

class PongGame:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)

        self.img_background = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\Background.png')
        self.img_game_over = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\gameOver.png')
        self.img_ball = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\Ball.png', cv2.IMREAD_UNCHANGED)
        self.img_bat1 = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\bat1.png', cv2.IMREAD_UNCHANGED)
        self.img_bat2 = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Resources\bat2.png', cv2.IMREAD_UNCHANGED)

        self.ball_pos = [100, 100]
        self.speed_x = 20
        self.speed_y = 20
        self.score = [0, 0]

        self.game_over = False
        self.countdownFlag = False
        self.detector = HandDetector(detectionCon=0.8, maxHands=2)

        # Powerup variables
        self.powerup_active = False
        self.powerup_timer = 0
        self.powerup_hand = None
        self.powerup_timer2 = 0

    def reset(self):
        self.ball_pos = [100, 100]
        self.speed_x = 25
        self.speed_y = 25
        self.score = [0, 0]
        self.game_over = False
        self.countdownFlag = False
        self.powerup_active = False
        self.powerup_timer = 0
        self.powerup_hand = None
        self.powerup_timer2 = 0
        self.img_game_over = cv2.imread(r'/CV_Pong_Game/Resources/gameOver.png')

    def countdown(self, img):
        for i in range(3, 0, -1):
            img_copy = img.copy()
            cv2.putText(img_copy, str(i), (600, 360), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 10)
            cv2.imshow('img', img_copy)
            cv2.waitKey(1000)

    def draw_powerup(self, img):
        if not self.powerup_active and random.randint(0, 120) < 5:  # 5% chance of powerup spawn
            self.powerup_x = random.randint(100, 1100)  # Random X-coordinate for powerup
            self.powerup_y = random.randint(100, 400)  # Random Y-coordinate for powerup
            self.powerup_active = True
            self.powerup_timer = 200  # Powerup duration in frames

        if self.powerup_active:
            radius = 25  # Radius of the powerup circle
            cv2.circle(img, (self.powerup_x, self.powerup_y), radius, (0, 255, 0), -1)  # Draw powerup circle
            if self.powerup_x - radius < self.ball_pos[0] < self.powerup_x + radius and self.powerup_y - radius < self.ball_pos[1] < self.powerup_y + radius:
                self.powerup_active = False
                self.powerup_timer2 = 100
                self.powerup_timer = 0
                if self.ball_pos[0] > 0:
                    self.powerup_hand = 'Right'
                else:
                    self.powerup_hand = 'Left'

        if self.powerup_timer:
            self.powerup_timer -= 1  # Decrement powerup timer
        if self.powerup_timer2:
            self.powerup_timer2 -= 1  # Decrement powerup timer
        else:
            self.powerup_hand = None
        if self.powerup_timer <= 0:
            self.powerup_active = False

        return img

    def draw_bats(self, img, hands):
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = self.img_bat1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == 'Left':
                if self.powerup_hand == 'Left':
                    img = cvzone.overlayPNG(img, self.img_bat1, (59, y1))
                    img = cvzone.overlayPNG(img, self.img_bat1, (59, y1 + (h1 - 30)))
                    if 59 - 10 < self.ball_pos[0] < 59 + w1 and y1 - (h1 // 2) < self.ball_pos[1] < y1 + (h1 * 2):
                        self.speed_x *= -1
                        self.ball_pos[0] += 20
                        self.score[0] += 1
                else:
                    img = cvzone.overlayPNG(img, self.img_bat1, (59, y1))
                    if 59 - 10 < self.ball_pos[0] < 59 + w1 and y1 - (h1 // 2) < self.ball_pos[1] < y1 + (h1 // 2):
                        self.speed_x *= -1
                        self.ball_pos[0] += 20
                        self.score[0] += 1

            if hand['type'] == 'Right':
                if self.powerup_hand == 'Right':
                    img = cvzone.overlayPNG(img, self.img_bat2, (1195, y1))
                    img = cvzone.overlayPNG(img, self.img_bat2, (1195, y1 + (h1 - 30)))
                    if 1120 < self.ball_pos[0] < 1170 + w1 and y1 - (h1 // 2) < self.ball_pos[1] < y1 + (h1 * 2):
                        self.speed_x *= -1
                        self.ball_pos[0] -= 20
                        self.score[1] += 1
                else:
                    img = cvzone.overlayPNG(img, self.img_bat2, (1195, y1))
                    if 1120 < self.ball_pos[0] < 1170 + w1 and y1 - (h1 // 2) < self.ball_pos[1] < y1 + (h1 // 2):
                        self.speed_x *= -1
                        self.ball_pos[0] -= 20
                        self.score[1] += 1

        return img

    def play_game(self, cam):
        self.cam = cam
        while True:
            success, img = self.cam.read()
            img = cv2.flip(img, 1)
            img_raw = img.copy()

            hands, img = self.detector.findHands(img, flipType=False)
            img = cv2.addWeighted(img, 0.2, self.img_background, 0.8, 0)

            if not self.countdownFlag:
                self.countdown(img)
                self.countdownFlag = True  # To ensure countdown only happens once

            img = self.draw_powerup(img)

            if hands:
                img = self.draw_bats(img, hands)

            if self.ball_pos[1] >= 500 or self.ball_pos[1] <= 10:
                self.speed_y *= -1

            if self.ball_pos[0] < 10 or self.ball_pos[0] > 1200:
                self.game_over = True

            if self.game_over:
                img = self.img_game_over
                cv2.putText(img, str(max(self.score[0], self.score[1])).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 3, (200, 0, 200), 5)
            else:
                self.ball_pos[0] += self.speed_x
                self.ball_pos[1] += self.speed_y

                cv2.putText(img, str(self.score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
                cv2.putText(img, str(self.score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

                img = cvzone.overlayPNG(img, self.img_ball, self.ball_pos)

            img[580:700, 20:233] = cv2.resize(img_raw, (213, 120))

            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('r'):
                self.reset()
            elif key == ord('q'):
                break

if __name__ == "__main__":
    game = PongGame()
    game.play_game(None)
