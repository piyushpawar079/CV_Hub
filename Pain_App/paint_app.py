import cv2
from cvzone.HandTrackingModule import HandDetector as hd
import os
import numpy as np
import math
import mysql.connector

class VirtualPainter:
    def __init__(self, username=None):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)
        self.detector = hd()
        self.username = username

        self.folder = r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Images'
        self.header_images = [cv2.imread(f'{self.folder}/{img}') for img in os.listdir(self.folder)]
        self.header = self.header_images[0]

        self.icon_img = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\MenuIcon2.png',
                                   cv2.IMREAD_UNCHANGED)
        self.icon_img = cv2.resize(self.icon_img, (40, 40))

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
        self.show_options = False
        self.fill_option = ''
        self.lm_list = []
        self.fill_type = None
        self.fill_start_angle = 0
        self.fill_end_angle = 0
        self.canvas_states = []
        self.max_states = 70  # Maximum number of states to store
        self.undo_button_active = False
        self.current_color = self.color1

        self.circle_x1, self.circle_y1, self.radius = 0, 0, 0
        self.line_start, self.line_end = (0, 0), (0, 0)
        self.img_canvas = np.zeros((720, 1280, 3), np.uint8)

        self.brush_size = 15
        self.min_brush_size = 5
        self.max_brush_size = 50

        self.slider_center = (130, 180)  # Center position of the circular slider
        self.slider_radius = 50  # Radius of the circular slider

        self.dropdown_button_active = False
        self.dropdown_options = ["Save", "Exit"]
        self.show_menu = False

    def draw_brush_slider(self, img):
        cv2.rectangle(img, (10, 130), (260, 160), (200, 200, 200), -1)
        cv2.rectangle(img, (10, 130), (10 + int(250 * (self.brush_size - self.min_brush_size) / (self.max_brush_size - self.min_brush_size)), 160), (0, 255, 0), -1)
        cv2.putText(img, f"Brush Size: {self.brush_size}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def adjust_brush_size(self, x):
        if 10 <= x <= 260:
            self.brush_size = int(self.min_brush_size + (x - 10) * (self.max_brush_size - self.min_brush_size) / 250)
            self.brush_thickness = self.brush_size

    def save_canvas_state(self):
        if len(self.canvas_states) >= self.max_states:
            self.canvas_states.pop(0)
        self.canvas_states.append(self.img_canvas.copy())
        self.undo_button_active = True

    def draw_menu_button(self, background, overlay, x, y):
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

        bg_h, bg_w, bg_channels = background.shape
        ol_h, ol_w, ol_channels = overlay.shape

        # Ensure the overlay is within the bounds of the background image
        if x + ol_w > bg_w or y + ol_h > bg_h:
            raise ValueError("Overlay image goes out of bounds of the background image.")

        # Get the region of interest (ROI) from the background image
        roi = background[y:y + ol_h, x:x + ol_w]

        # Convert overlay image to have an alpha channel if it doesn't already
        if overlay.shape[2] == 3:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
        if background.shape[2] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

        # Separate the alpha channel from the overlay image
        overlay_img = overlay[:, :, :3]
        alpha_mask = overlay[:, :, 3] / 255.0

        # Blend the ROI and the overlay image
        for c in range(0, 3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + overlay_img[:, :, c] * alpha_mask

        # Replace the original ROI in the background image with the blended ROI
        background[y:y + ol_h, x:x + ol_w] = roi

        return background

    def draw(self, cam=None):
        if cam:
            self.cam = cam
        while True:
            _, img = self.cam.read()
            if _:
                img = cv2.flip(img, 1)
                hands, img = self.detector.findHands(img, flipType=False)

                self.draw_undo_button(img)
                self.draw_brush_slider(img)

                if self.show_options:
                    img = self.draw_options(img)

                if hands:
                    self.lm_list = self.detector.findPosition(img)
                    if self.lm_list:
                        if self.process_hand_gestures(img, hands):
                            break

                        if 1000 < self.lm_list[8][1] < 1180 and 50 < self.lm_list[8][2] < 140 and not self.show_menu:
                            self.show_menu = True
                        if self.show_menu:
                            img = self.draw_menu(img)

                        if self.fill_type:
                            cv2.ellipse(img, (self.circle_x1, self.circle_y1), (self.radius, self.radius),
                                        0, self.fill_start_angle, self.fill_end_angle, self.color2, 2)

                img_gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
                _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(img, img_inv)
                img = cv2.bitwise_or(img, self.img_canvas)

                img[:104, :1007] = self.header

                img = self.draw_menu_button(img, self.icon_img, 1100, 80)

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
            self.xp, self.yp = 0, 0
            if 1090 < x1 < 1180 and 10 < y1 < 60 and self.undo_button_active:
                # print(f"Attempting to undo. Button active: {self.undo_button_active}")
                self.undo()
                cv2.waitKey(10)
            elif 130 < y1 < 160:
                self.adjust_brush_size(x1)
            elif x1 < 1000:
                self.show_menu = False
                self.select_tool(x1, y1, x2, y2, img)
        elif fingers[1] and not fingers[2]:
            if self.show_options:
                self.select_fill_option(x1, y1)
            elif self.fill_type:
                self.select_fill_area(x1, y1, img)
            elif self.show_menu:
                if 1000 < x1 < 1180 and 70 < y1 < 200:
                    # if 100 < y1 < 140:  # Save
                    #     print('save')
                    #     self.save_screenshot(img)
                    if 150 < y1 < 200:  # Exit
                        print('exit')
                        cv2.destroyAllWindows()
                        return 1
            else:
                self.draw_on_canvas(img, hands)
        elif not fingers[1] and not fingers[2]:
            if self.fill_type:
                self.apply_selected_fill()
                self.save_canvas_state()
                self.undo_button_active = True
                # print("Canvas state saved, undo button activated")
            self.xp, self.yp = 0, 0
        else:
            self.xp, self.yp = 0, 0

    def save_screenshot(self, img):
        cv2.imwrite(f'Output.jpg', img)

    def draw_menu(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (1000, 120), (1250, 210), (50, 50, 50), -1)  # Options background
        # cv2.putText(overlay, "Save", (1020, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Exit", (1020, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Blend the overlay with the original image
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        return img

    def select_fill_option(self, x, y):
        if 900 < x < 1250:
            if 100 < y < 200:
                # print("Full circle fill selected")
                self.fill_type = "full"
            elif 200 < y < 300:
                # print("Half circle fill selected")
                self.fill_type = "half"
            elif 300 < y < 400:
                # print("Quarter circle fill selected")
                self.fill_type = "quarter"

            if self.fill_type:
                self.show_options = False

    def select_fill_area(self, x, y, img):
        dx = x - self.circle_x1
        dy = self.circle_y1 - y  # Invert y-axis
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        if self.fill_type == "full":
            self.fill_start_angle = 0
            self.fill_end_angle = 360
        elif self.fill_type == "half":
            self.fill_start_angle = angle
            self.fill_end_angle = (angle + 180) % 360
        elif self.fill_type == "quarter":
            self.fill_start_angle = angle
            self.fill_end_angle = (angle + 90) % 360

        # Create a separate mask for the preview
        preview_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.ellipse(preview_mask, (self.circle_x1, self.circle_y1), (self.radius, self.radius),
                    0, self.fill_start_angle, self.fill_end_angle, 255, -1)

        # Create an overlay with the preview color
        preview_color = np.array(self.color2, dtype=np.uint8)
        overlay = np.full(img.shape, preview_color, dtype=np.uint8)

        # Apply the overlay only to the masked area
        mask_3channel = cv2.merge([preview_mask, preview_mask, preview_mask])
        overlay_area = cv2.bitwise_and(overlay, mask_3channel)
        img_area = cv2.bitwise_and(img, cv2.bitwise_not(mask_3channel))

        # Blend the overlay with the original image
        result = cv2.add(img_area, overlay_area)
        cv2.addWeighted(img, 0.5, result, 0.5, 0, dst=img)

    def apply_selected_fill(self):
        if self.fill_type:
            mask = np.zeros(self.img_canvas.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, (self.circle_x1, self.circle_y1), (self.radius, self.radius),
                        0, self.fill_start_angle, self.fill_end_angle, 255, -1)
            self.img_canvas[mask == 255] = self.color2
            self.fill_type = None  # Reset fill type after applying

    def apply_fill(self, start_angle, end_angle):
        mask = np.zeros(self.img_canvas.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (self.circle_x1, self.circle_y1), (self.radius, self.radius),
                    0, start_angle, end_angle, 255, -1)
        self.img_canvas[mask == 255] = self.color2

    def draw_options(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (900, 100), (1250, 400), (50, 50, 50), -1)  # Options background
        cv2.putText(overlay, "Fill full circle", (920, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Fill half circle", (920, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Fill quarter circle", (920, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Blend the overlay with the original image
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        return img

    def select_tool(self, x1, y1, x2, y2, img):
        if y1 < 130:
            # self.header = self.header_images[0]
            if 10 < x1 < 100:
                self.header = self.header_images[0]
                self.color1 = (255, 192, 203)
                self.color2 = (0, 0, 255)
                self.color3 = (255, 192, 203)
                self.selected = 'brush1'
            elif 200 < x1 < 300:
                self.header = cv2.resize(self.header_images[1], (1007, 104), interpolation=cv2.INTER_AREA)
                self.color1 = (0, 0, 255)
                self.color2 = (0, 0, 255)
                self.color3 = (0, 0, 255)
                self.selected = 'brush2'
            elif 450 < x1 < 550:
                self.select_circle()
            elif 600 < x1 < 700:
                self.select_line()
            elif 800 < x1 < 900:
                self.color1 = (0, 0, 0)
                self.color2 = (0, 0, 255)
                self.selected = 'eraser'

        cv2.line(img, (x1, y1), (x2, y2), self.color3, 3)

    def select_circle(self):
        self.color2 = (0, 0, 0)
        self.color1 = (0, 0, 0)
        self.color3 = (255, 0, 255)
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

        # Check if the finger was just lowered
        if self.xp == 0 and self.yp == 0:
            self.xp, self.yp = x1, y1

        # Calculate the distance between current and previous point
        distance = ((x1 - self.xp) ** 2 + (y1 - self.yp) ** 2) ** 0.5

        # If the distance is too large, assume the finger was lifted and reset the previous point
        if distance > 50:  # You may need to adjust this threshold
            self.xp, self.yp = x1, y1

        if self.selected == 'brush1' or self.selected == 'brush2':
            self.draw_line(x1, y1, img)
        elif self.selected == 'eraser':
            self.draw_eraser(x1, y1, img)
        elif self.selected == 'circle':
            self.draw_circle(x1, y1, img, hands)
        elif self.selected == 'line':
            self.draw_line_shape(x1, y1, img, hands)

        self.xp, self.yp = x1, y1

        self.save_canvas_state()
        # self.undo_button_active = True
        # print("Canvas state saved after drawing")

    def draw_undo_button(self, img):
        if self.undo_button_active:
            cv2.rectangle(img, (1090, 10), (1180, 60), (0, 255, 0), -1)
            cv2.putText(img, "Undo", (1105, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.rectangle(img, (1090, 10), (1180, 60), (200, 200, 200), -1)
            cv2.putText(img, "Undo", (1105, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    def undo(self):
        if len(self.canvas_states) > 0:
            self.img_canvas = self.canvas_states.pop()
        if len(self.canvas_states) == 0:
            self.undo_button_active = False

    def draw_line(self, x1, y1, img):
        cv2.line(img, (self.xp, self.yp), (x1, y1), self.color1, self.brush_thickness)
        cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.color1, self.brush_thickness)

    def draw_eraser(self, x1, y1, img):
        cv2.line(img, (self.xp, self.yp), (x1, y1), self.color1, self.eraser_thickness)
        cv2.line(self.img_canvas, (self.xp, self.yp), (x1, y1), self.color1, self.eraser_thickness)

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
                self.show_options = True
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
    painter = VirtualPainter(username='OM')
    painter.draw()