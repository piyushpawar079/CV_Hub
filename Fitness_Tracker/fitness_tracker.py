import cv2
import numpy as np
import time
from HandsGestureDetector import HandDetector
from Fitness_Tracker.PoseModule import poseDetector

class ArmCurlsCounter:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.detector = poseDetector()
        self.count = 0
        self.dir = 0
        self.pTime = 0
        self.hands_Detector = HandDetector()
        self.active_arm = 'right'  # Default to right arm
        self.button_left = {'x1': 50, 'y1': 50, 'x2': 200, 'y2': 100}
        self.button_right = {'x1': 250, 'y1': 50, 'x2': 400, 'y2': 100}
        self.f = 0
        self.last_switch_time = 0
        self.switch_delay = 1.0

    def process_frame(self, img):
        img = self.detector.findPose(img, False)
        lmList = self.detector.findPosition(img, False)
        if len(lmList) != 0:
            if self.active_arm == 'right':
                shoulder, elbow, wrist = 15, 13, 11
                angle = self.detector.findAngle(img, shoulder, elbow, wrist)
            else:
                shoulder, elbow, wrist = 12, 14, 16
                angle = self.detector.findAngle(img, shoulder, elbow, wrist)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))
            color = self.update_count(per)
            self.draw_ui(img, per, bar, color)
        return img

    def update_count(self, per):
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if self.dir == 0:
                self.count += 0.5
                self.dir = 1
        if per == 0:
            color = (0, 255, 0)
            if self.dir == 1:
                self.count += 0.5
                self.dir = 0
        return color

    def draw_ui(self, img, per, bar, color):
        # Progress bar
        cv2.rectangle(img, (1100, 100), (1175, 650), (200, 200, 200), 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1120, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Count display
        cv2.rectangle(img, (0, 450), (250, 720), (245, 117, 16), cv2.FILLED)
        cv2.putText(img, str(int(self.count)), (45, 670), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 25)
        cv2.putText(img, "REPS", (40, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)

        # Arm selection buttons
        self.draw_button(img, self.button_left, 'Left Arm', self.active_arm == 'left')
        self.draw_button(img, self.button_right, 'Right Arm', self.active_arm == 'right')

        if time.time() - self.last_switch_time < self.switch_delay:
            cv2.putText(img, "Switching...", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def draw_button(self, img, button, text, is_active):
        color = (0, 255, 0) if is_active else (200, 200, 200)
        cv2.rectangle(img, (button['x1'], button['y1']), (button['x2'], button['y2']), color, cv2.FILLED)
        cv2.putText(img, text, (button['x1'] + 10, button['y1'] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def show_fps(self, img):
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (1100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def handle_click(self, lmlist, fingers):
        current_time = time.time()
        switch_arm = False

        if self.button_left['x1'] < lmlist[8][1] < self.button_left['x2'] and self.button_left['y1'] < lmlist[8][2] < self.button_left['y2']:
            switch_arm = True
        elif self.button_right['x1'] < lmlist[8][1] < self.button_right['x2'] and self.button_right['y1'] < lmlist[8][2] < self.button_right['y2']:
            switch_arm = True
        elif fingers == [1, 0, 0, 0, 1] or fingers == [0, 0, 0, 0, 1]:
            switch_arm = True

        if switch_arm and (current_time - self.last_switch_time) >= self.switch_delay:
            self.f += 1
            self.last_switch_time = current_time
            if self.f % 2:
                self.active_arm = 'left'
            else:
                self.active_arm = 'right'

        if 900 < lmlist[8][1] < 1000 and 50 < lmlist[8][2] < 90:
            return 1

    def draw_rectangle_with_text(self, image, top_left, bottom_right, text):
        # Draw the rectangle
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), -1)

        # Add a border around the rectangle
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 2)

        # Calculate the position for the text
        text_position = (top_left[0] + 10, top_left[1] + 30)

        # Draw the text with a shadow
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, text, (text_position[0] + 2, text_position[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        return image

    def run(self, cam=None):
        if cam:
            self.cap = cam
        cv2.namedWindow("Arm Curls Counter")

        while True:
            success, img = self.cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            hands, img = self.hands_Detector.findHands(img, draw=True, flipType=False)
            if hands:
                lmlist = self.hands_Detector.findPosition(img)
                if lmlist:
                    fingers = self.hands_Detector.fingersUp(hands[0])
                    if self.handle_click(lmlist, fingers):
                        cv2.destroyAllWindows()
                        break
            img = self.process_frame(img)
            img = self.show_fps(img)
            img = self.draw_rectangle_with_text(img, (900, 50),(1000, 90), 'BACK')
            cv2.imshow("Arm Curls Counter", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    arm_curls_counter = ArmCurlsCounter()
    arm_curls_counter.run()