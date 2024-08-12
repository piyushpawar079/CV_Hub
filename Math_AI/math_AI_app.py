import cv2
import numpy as np
import google.generativeai as genai
from cvzone.HandTrackingModule import HandDetector as hd
from PIL import Image
import textwrap

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
        self.response_rectangle = (10, 10, 400, 700)

    def initialize_camera(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 1280)
        cam.set(4, 720)
        return cam

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
            if 900 < lmList[8][1] < 1000 and 50 < lmList[8][2] < 90:
                self.over = True
        elif fingers == [0, 1, 1, 1, 1] or fingers == [1, 1, 1, 1, 1]:
            self.canvas = np.zeros_like(img)
            self.output_text = ''
        # elif fingers == [0, 0, 0, 0, 0]:
        #     self.over = True

    def send_to_ai(self, canvas, fingers):
        if fingers == [0, 0, 0, 0, 1] or fingers == [1, 0, 0, 0, 1]:
            resized_canvas = cv2.resize(canvas, (512, 512))
            pil_img = Image.fromarray(resized_canvas)
            response = self.model.generate_content(["solve this math problem: ", pil_img, ". If the question is complex please explain in detail"])
            return response.text
        return ''

    def draw_response_rectangle(self, image):
        x, y, w, h = self.response_rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

        if self.output_text:
            # Split the text into sentences
            sentences = self.output_text.split('.')
            # Remove empty sentences and add the period back
            sentences = [sentence.strip() + '.' for sentence in sentences if sentence.strip()]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 1
            line_height = 30
            max_width = w - 20  # Maximum width for text, leaving some padding

            current_y = y + 30
            for sentence in sentences:
                # Wrap each sentence
                wrapped_lines = textwrap.wrap(sentence, width=30)  # Adjust width as needed
                for line in wrapped_lines:
                    # Check if we've reached the bottom of the rectangle
                    if current_y + line_height > y + h:
                        break

                    cv2.putText(image, line, (x + 10, current_y), font, font_scale, (255, 255, 255), font_thickness)
                    current_y += line_height

                # Add an extra line break after each sentence
                current_y += line_height // 2

                # Check if we've reached the bottom of the rectangle
                if current_y > y + h:
                    break

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

    def run_app(self, cap=None):
        if cap is not None:
            self.cam = cap
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
                new_output = self.send_to_ai(self.canvas, info[0])
                if new_output:
                    self.output_text = ''
                    self.output_text = new_output

            combined = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)

            self.draw_response_rectangle(combined)
            combined = self.draw_rectangle_with_text(combined, (900, 50), (1000, 90), 'BACK')

            cv2.imshow('Hand Gesture AI', combined)
            key = cv2.waitKey(1)
            if key == ord('q') or self.over:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    api_key = 'AIzaSyBLoq2qPnvxqfGfyqZb2ifo202nkcPPKKA'
    hand_gesture_ai = HandGestureAI(api_key)
    hand_gesture_ai.run_app()
