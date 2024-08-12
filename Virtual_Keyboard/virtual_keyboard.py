import cv2
import numpy as np
import cvzone
import time
from PIL import ImageFont, ImageDraw, Image
import os
# from spellchecker import SpellChecker
from pynput.keyboard import Controller
from cvzone.HandTrackingModule import HandDetector


class VirtualKeyboard:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)

        self.detector = HandDetector()
        self.keyboard = Controller()

        self.keys_en = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
            ["SHIFT", "SPACE", "BACK", "CAPS", "LANG", "CLEAR", "ENTER"]
        ]

        self.keys_mr = [
            ["क", "ख", "ग", "घ", "च", "छ", "ज", "झ", "ञ", "ट"],
            ["ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प"],
            ["फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष"],
            ["SHIFT", "SPACE", "BACK", "CAPS", "LANG", "CLEAR", "ENTER"]
        ]

        self.current_keys = self.keys_en
        self.final_text = ''
        self.prev_len = 0

        font_path = "../Resources/static/NotoSansDevanagari-Regular.ttf"
        if not os.path.isfile(font_path):
            raise FileNotFoundError(
                f"Font file '{font_path}' not found. Please ensure the file exists and the path is correct.")
        self.font = ImageFont.truetype(font_path, 32)

        self.shift = False
        self.caps = False
        self.debounce_time = 1
        self.last_press_time = time.time()

        # self.spell = SpellChecker('en')

        self.button_list = self.create_button_list()

        self.mode = "normal"  # Add this line to track the current mode
        self.save_button = Button([1100, 100], "SAVE", [100, 80])

    def enter_name(self, cam=None):
        self.mode = "name_entry"
        self.final_text = ""
        try:
            if cam:
                self.cam = cam
            while True:
                success, img = self.cam.read()
                img = cv2.flip(img, 1)

                hands, img = self.detector.findHands(img, flipType=False)
                lmlist1 = self.detector.findPosition(img)

                img = self.draw_all(img)

                if len(lmlist1):
                    for button in self.button_list:
                        x, y = button.pos
                        w, h = button.size
                        name = self.check_hand_position(img, button, hands)
                        if name:
                            cv2.destroyWindow('img')
                            return name

                img = draw_text_with_pil(img, f"Enter name: {self.final_text}", (50, 50), self.font, (0, 0, 255))

                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        # finally:
            # cv2.destroyWindow('img')
        return self.final_text.strip()

    def add_content_to_file(self, cam):
        self.mode = "file_entry"
        self.final_text = ""
        if cam:
            self.cam = cam
        while True:
            success, img = self.cam.read()
            img = cv2.flip(img, 1)

            hands, img = self.detector.findHands(img, flipType=False)
            lmlist1 = self.detector.findPosition(img)

            img = self.draw_all(img)

            # Draw the save button
            x, y = self.save_button.pos
            w, h = self.save_button.size
            cv2.rectangle(img, self.save_button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
            self.save_button.render_text(img, self.font)

            if len(lmlist1) and hands:
                for button in self.button_list + [self.save_button]:
                    n = self.check_hand_position(img, button, hands)
                    if n:
                        cv2.destroyAllWindows()
                        return

            img = draw_text_with_pil(img, self.final_text, (50, 50), self.font, (0, 0, 255))

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def create_button_list(self):
        button_list = []
        c = 0
        for i in range(len(self.current_keys)):
            for j, key in enumerate(self.current_keys[i]):
                if key in ["SPACE", "BACK", "CAPS", "LANG", "CLEAR", "ENTER"]:
                    button_list.append(Button([100 * j + 80 + c, 100 * i + 150], key, [100, 80]))
                    c += 30
                elif key in ['SHIFT']:
                    button_list.append(Button([100 * j + 50, 100 * i + 150], key, [100, 80]))
                else:
                    button_list.append(Button([100 * j + 50, 100 * i + 150], key))
        return button_list

    def draw_all(self, img):
        img_new = np.zeros_like(img, np.uint8)
        for button in self.button_list:
            x, y = button.pos
            cvzone.cornerRect(img_new, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
            cv2.rectangle(img_new, button.pos, (x + button.size[0], y + button.size[1]), (255, 0, 255), cv2.FILLED)
            button.render_text(img_new, self.font)
        out = cv2.addWeighted(img, 0.5, img_new, 0.5, 0)
        return out

    def handle_button_press(self, button_text):
        if button_text == "LANG":
            self.switch_language()
        elif button_text == "SHIFT":
            self.shift = not self.shift
        elif button_text == "CAPS":
            self.caps = not self.caps
        elif button_text == "BACK":
            self.final_text = self.final_text[:-1]
        elif button_text == "CLEAR":
            self.final_text = ''
            self.prev_len = 0
        elif button_text == "SPACE":
            self.final_text += " "
        elif button_text == "ENTER":
            self.final_text += '\n'
        else:
            if self.shift:
                button_text = button_text.upper()
            elif self.caps:
                button_text = button_text.lower()
            self.final_text += button_text

    def check_hand_position(self, img, button, hands):
        x, y = button.pos
        w, h = button.size
        lmlist1 = self.detector.findPosition(img, 0)
        if len(hands) == 2:
            lmlist2 = self.detector.findPosition(img, 1)
            for lmlist in [lmlist1, lmlist2]:
                if (x < lmlist[8][1] < x + w and y < lmlist[8][2] < y + h) or (
                        x < lmlist[12][1] < x + w and y < lmlist[12][2] < y + h):
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 255), cv2.FILLED)
                    l, _, _ = self.detector.findDistance((lmlist[8][1], lmlist[8][2]), (lmlist[12][1], lmlist[12][2]),
                                                         img)

                    if self.mode == 'file_entry':
                        if int(l) < 45:
                            if time.time() - self.last_press_time > self.debounce_time:
                                self.last_press_time = time.time()
                                if button.text == "SAVE":
                                    with open("output.txt", "w", encoding="utf-8") as f:
                                        f.write(self.final_text)
                                    self.mode = "normal"
                                    return 1
                                else:
                                    self.handle_button_press(button.text)

                    if self.mode == 'name_entry':
                        if int(l) < 45:
                            if time.time() - self.last_press_time > self.debounce_time:
                                self.last_press_time = time.time()
                                if button.text == "ENTER":
                                    self.mode = "normal"
                                    return self.final_text.strip()
                                else:
                                    self.handle_button_press(button.text)

                    if int(l) < 45 and time.time() - self.last_press_time > self.debounce_time:
                        self.last_press_time = time.time()
                        self.handle_button_press(button.text)

        else:
            if (x < lmlist1[8][1] < x + w and y < lmlist1[8][2] < y + h) or (
                    x < lmlist1[12][1] < x + w and y < lmlist1[12][2] < y + h):
                cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 255), cv2.FILLED)
                l, _, _ = self.detector.findDistance((lmlist1[8][1], lmlist1[8][2]), (lmlist1[12][1], lmlist1[12][2]),
                                                     img)
                if self.mode == 'file_entry':
                    if int(l) < 45:
                        if time.time() - self.last_press_time > self.debounce_time:
                            self.last_press_time = time.time()
                            if button.text == "SAVE":
                                with open("output.txt", "w", encoding="utf-8") as f:
                                    f.write(self.final_text)
                                self.mode = "normal"
                                return 1
                            else:
                                self.handle_button_press(button.text)

                if self.mode == 'name_entry':
                    if int(l) < 45:
                        if time.time() - self.last_press_time > self.debounce_time:
                            self.last_press_time = time.time()
                            if button.text == "ENTER":
                                self.mode = "normal"
                                print(self.final_text)
                                return self.final_text.strip()
                            else:
                                self.handle_button_press(button.text)

                if int(l) < 45 and time.time() - self.last_press_time > self.debounce_time:
                    self.last_press_time = time.time()
                    self.handle_button_press(button.text)

    def switch_language(self):
        self.current_keys = self.keys_mr if self.current_keys == self.keys_en else self.keys_en
        self.button_list = self.create_button_list()

    # def autocorrect(self):
    #     if self.final_text and self.final_text[-1] == " " and self.current_keys == self.keys_en:
    #         words = self.final_text.split()
    #         if len(words) > self.prev_len:
    #             last_word = words[-1]
    #             corrected_word = self.spell.correction(last_word)
    #             if corrected_word != last_word:
    #                 self.prev_len = len(words)
    #                 if self.caps:
    #                     corrected_word = corrected_word.upper()
    #                 words[-1] = corrected_word
    #                 self.final_text = ' '.join(words) + ' '

    def run(self):
        while True:
            success, img = self.cam.read()
            img = cv2.flip(img, 1)

            hands, img = self.detector.findHands(img, flipType=False)
            lmlist1 = self.detector.findPosition(img)

            img = self.draw_all(img)

            if len(lmlist1) and hands:

                for button in self.button_list:
                    x, y = button.pos
                    w, h = button.size

                    self.check_hand_position(img, button, hands)

            # self.autocorrect()

            img = draw_text_with_pil(img, self.final_text, (50, 50), self.font, (0, 0, 255))

            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()


class Button:
    def __init__(self, pos, text, size=[60, 60]):
        self.pos = pos
        self.size = size
        self.text = text
        self.rendered_text = None

    def render_text(self, img, font):
        if self.rendered_text is None:
            self.rendered_text = draw_text_with_pil(np.zeros_like(img, np.uint8), self.text,
                                                    (self.pos[0] + 10, self.pos[1] + 20), font, (255, 255, 255))
        img[self.pos[1]:self.pos[1] + self.size[1], self.pos[0]:self.pos[0] + self.size[0]] = cv2.addWeighted(
            img[self.pos[1]:self.pos[1] + self.size[1], self.pos[0]:self.pos[0] + self.size[0]], 0.5,
            self.rendered_text[self.pos[1]:self.pos[1] + self.size[1], self.pos[0]:self.pos[0] + self.size[0]], 0.5, 0
        )


def draw_text_with_pil(img, text, position, font, color):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return np.array(pil_img)


if __name__ == "__main__":
    virtual_keyboard = VirtualKeyboard()
    # virtual_keyboard.run()
    print(virtual_keyboard.add_content_to_file())