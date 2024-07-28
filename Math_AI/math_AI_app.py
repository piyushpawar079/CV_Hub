import cv2
import numpy as np
import google.generativeai as genai
from cvzone.HandTrackingModule import HandDetector as hd
from PIL import Image


class HandGestureAI:
    def __init__(self, api_key, model_name='gemini-1.5-flash'):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.prev_pos = None
        self.canvas = None
        self.output_text = ''
        self.detector = hd(maxHands=1)
        self.cam = self.initialize_camera()
        self.initialize_genai()
        self.over = False

    def initialize_camera(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 1280)
        cam.set(4, 720)
        # return cam

    def initialize_genai(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def get_hand_info(self, img):
        hands, img = self.detector.findHands(img, flipType=False)
        if hands:
            hand = hands[0]
            lmList = self.detector.findPosition(img)
            fingers = self.detector.fingersUp(hand)
            return fingers, lmList
        else:
            return None

    def draw(self, info, img):
        fingers, lmList = info
        current_pos = None

        if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
            current_pos = lmList[8][1], lmList[8][2]
            if self.prev_pos is None:
                self.prev_pos = current_pos
            cv2.line(self.canvas, self.prev_pos, current_pos, (255, 0, 255), 10)
            self.prev_pos = current_pos
        elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
            self.prev_pos = None
        elif fingers == [0, 1, 1, 1, 1] or fingers == [1, 1, 1, 1, 1]:
            self.canvas = np.zeros_like(img)
        elif fingers == [0, 0, 0, 0, 0]:
            self.over = True

    def send_to_ai(self, canvas, fingers):
        if fingers == [0, 1, 1, 1, 0] or fingers == [1, 1, 1, 1, 0]:
            pil_img = Image.fromarray(canvas)
            response = self.model.generate_content(["solve this math problem: ", pil_img])
            return response.text
        return ''

    def run_app(self, cam):
        self.cam = cam
        while True:
            success, img = self.cam.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            if self.canvas is None:
                self.canvas = np.zeros_like(img)
            info = self.get_hand_info(img)
            if info:
                self.draw(info, img)
                self.output_text = self.send_to_ai(self.canvas, info[0])

            combined = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)

            # Draw rectangle for AI output
            cv2.rectangle(combined, (10, 10, 600, 150), (0, 0, 0), -1)
            cv2.putText(combined, self.output_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Hand Gesture AI', combined)
            key = cv2.waitKey(1)
            if key == ord('q') or self.over:
                self.over = True
                break

        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    api_key = 'AIzaSyBLoq2qPnvxqfGfyqZb2ifo202nkcPPKKA'
    hand_gesture_ai = HandGestureAI(api_key)
    hand_gesture_ai.run_app()
