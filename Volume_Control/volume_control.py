import cv2
import time
import numpy as np
import HandsGestureDetector as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class VolumeControl:
    def __init__(self, wCam=1280, hCam=720):
        self.cap = self.initialize_camera(wCam, hCam)
        self.detector = htm.HandDetector(maxHands=1)
        self.volume, self.minVolume, self.maxVolume = self.initialize_volume_control()
        self.selected = 1
        self.pTime = 0
        self.volBar = 400
        self.vol = 0
        self.volPer = 0
        self.volbar1 = 150
        self.volbar2 = 157
        self.over = False

    @staticmethod
    def initialize_camera(wCam, hCam):
        cap = cv2.VideoCapture(0)
        cap.set(3, wCam)
        cap.set(4, hCam)
        return cap

    @staticmethod
    def initialize_volume_control():
        device = AudioUtilities.GetSpeakers()
        interface = device.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        minVolume = volume.GetVolumeRange()[0]
        maxVolume = volume.GetVolumeRange()[1]
        return volume, minVolume, maxVolume

    @staticmethod
    def draw_hand_landmarks(img, lmlist):
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return x1, y1, x2, y2, cx, cy

    @staticmethod
    def update_volume(length, minVolume, maxVolume):
        vol = np.interp(length, [50, 300], [minVolume, maxVolume])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = int(np.interp(length, [50, 300], [0, 100]) // 5) * 5
        return vol, volBar, volPer

    @staticmethod
    def display_options(img):
        cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), -1)
        cv2.rectangle(img, (300, 100), (400, 200), (0, 0, 0), -1)
        cv2.putText(img, '1', (130, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        cv2.putText(img, '2', (330, 160), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    @staticmethod
    def selected_option(fingers, lmList, selected):
        x, y = lmList[8][1], lmList[8][2]

        if fingers[1] and 100 < x < 200 and 100 < y < 200:
            selected = 0
        elif fingers[1] and 300 < x < 400 and 100 < y < 200:
            selected = 1
        return selected

    @staticmethod
    def display_volume_bar(img, volBar, volPer, selected, volbar1, volbar2):
        if not selected:
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 1)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (10, 470), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        else:
            cv2.rectangle(img, (150, 300), (950, 400), (0, 255, 0), 5)
            cv2.rectangle(img, (150, 300), (volbar1, 400), (0, 255, 0), -1)
            cv2.rectangle(img, (volbar1, 295), (volbar2, 405), (0, 255, 255), -1)
            cv2.putText(img, f'{0}%', (80, 368), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            cv2.putText(img, f'{100}%', (960, 365), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            cv2.putText(img, f'{int(volPer)} %', (volbar1 - 25, 450), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    @staticmethod
    def display_fps(img, fps):
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 0, 255), 3)

    # def toggle_mute(self):
    #     current_volume = self.volume.GetMasterVolumeLevel()
    #     if current_volume != self.minVolume:
    #         self.volume.SetMasterVolumeLevel(self.minVolume, None)
    #     else:
    #         self.volume.SetMasterVolumeLevel(self.maxVolume, None)

    # def change_brightness(level):
    #     sbc.set_brightness(level)

    def process_frame(self, img):
        img = cv2.flip(img, 1)
        hands, img = self.detector.findHands(img)
        description = ''
        self.display_options(img)

        if hands:
            lmlist = self.detector.findPosition(img)
            fingers = self.detector.fingersUp(hands[0])

            if fingers == [0, 1, 1, 1, 1]:
                self.over = True

            elif fingers[1] and fingers[2] and fingers[3]:
                description = 'Selection Mode:- You can select any of the two given options to change the volume.'
                self.selected = self.selected_option(fingers, lmlist, self.selected)

            else:
                if not self.selected and fingers[1] and not fingers[2]:
                    description = 'Change Mode 1:- You can use your index finger and thumb to change the volume by increasing or decreasing the distance between them.'
                    x1, y1, x2, y2, cx, cy = self.draw_hand_landmarks(img, lmlist)
                    length = math.hypot(x2 - x1, y2 - y1)
                    self.vol, self.volBar, self.volPer = self.update_volume(length, self.minVolume, self.maxVolume)
                    if fingers[4]:
                        self.volume.SetMasterVolumeLevel(self.vol, None)
                elif self.selected and fingers[1] and fingers[2]:
                    description = 'Change Mode 2:- You can use your index finger and middle finger to change the volume by placing your index finger anywhere inside the rectangle.'
                    x11, y11 = lmlist[8][1], lmlist[8][2]
                    if fingers[1] and fingers[2] and 300 <= y11 <= 400:
                        x11 = int(x11)
                        self.vol = np.interp(self.volbar1, [150, 950], [self.minVolume, self.maxVolume])
                        self.volbar1 = int(np.interp(x11, [150, 950], [150, 950]))
                        self.volbar2 = self.volbar1 + 7
                        self.volPer = int(np.interp(self.volbar1, [150, 950], [0, 100]) // 5) * 5
                        self.volume.SetMasterVolumeLevel(self.vol, None)
                # elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
                #     description = 'Brightness Control Mode:- You can use your index finger and ring finger to change the brightness by increasing or decreasing the distance between them.'
                #     x1, y1, x2, y2, cx, cy = self.draw_hand_landmarks(img, lmlist)
                #     length = math.hypot(x2 - x1)
                #     brightness = np.interp(length, [50, 300], [0, 100])
                #     change_brightness(int(brightness))
                # elif fingers[4] and not any(fingers[:4]):
                #     description = 'Mute/Unmute Mode:- You can use your pinky finger to toggle mute and unmute.'
                #     self.toggle_mute()
                else:
                    description = 'There is nothing assigned to this gesture.'

        # self.draw_text_within_rectangle(img, description, (500, 50), (1100, 200))
        self.display_volume_bar(img, self.volBar, self.volPer, self.selected, self.volbar1, self.volbar2)

        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        self.display_fps(img, fps)

        return img

    # @staticmethod
    # def draw_text_within_rectangle(image, text, rect_top_left, rect_bottom_right, font=cv2.FONT_HERSHEY_SIMPLEX,
    #                                font_scale=1, font_color=(255, 255, 255), font_thickness=2):
    #     # Define the rectangle
    #     cv2.rectangle(image, rect_top_left, rect_bottom_right, (0, 255, 0), 2)
    #
    #     # Calculate the width of the rectangle
    #     rect_width = rect_bottom_right[0] - rect_top_left[0]
    #     rect_height = rect_bottom_right[1] - rect_top_left[1]
    #
    #     # Split text into words
    #     words = text.split(' ')
    #     current_line = ''
    #     y0, dy = rect_top_left[1] + 30, 30
    #
    #     for word in words:
    #         # Check if adding the next word will go out of the rectangle's width
    #         test_line = current_line + word + ' '
    #         text_size, _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
    #
    #         if text_size[0] > rect_width:
    #             # Write the current line on the image
    #             cv2.putText(image, current_line, (rect_top_left[0] + 10, y0), font, font_scale, font_color,
    #                         font_thickness, lineType=cv2.LINE_AA)
    #             current_line = word + ' '
    #             y0 += dy
    #
    #             # Check if we are exceeding the height of the rectangle
    #             if (y0 + dy - rect_top_left[1]) > rect_height:
    #                 break
    #         else:
    #             current_line = test_line
    #
    #     # Write the last line
    #     cv2.putText(image, current_line, (rect_top_left[0] + 10, y0), font, font_scale, font_color, font_thickness,
    #                 lineType=cv2.LINE_AA)

    def run(self, cam):
        self.cap = cam
        while True:
            success, img = self.cap.read()
            if not success:
                break

            img = self.process_frame(img)
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == ord('q') or self.over:
                self.over = False
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vc = VolumeControl()
    vc.run()
