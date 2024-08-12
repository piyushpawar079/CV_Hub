import cv2
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector as hd
from Volume_Control.volume_control import VolumeControl
from Pain_App.paint_app import VirtualPainter
from Presentation_App.presentation_app import PresentationController
from Pong_Game.pong_app import PongGame
from Virtual_Mouse.virtual_mouse import VirtualMouse
from Math_AI.math_AI_app import HandGestureAI
from Virtual_Keyboard.virtual_keyboard import VirtualKeyboard
from Fitness_Tracker.fitness_tracker import ArmCurlsCounter
import os

class CVApp:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 1280)
        self.cam.set(4, 720)

        api_key = 'AIzaSyBLoq2qPnvxqfGfyqZb2ifo202nkcPPKKA'

        self.detector = hd(maxHands=1)
        self.vol_control = VolumeControl()
        self.vir_paint = VirtualPainter()
        self.present_app = PresentationController()
        self.vir_mouse = VirtualMouse()
        self.pong_game = PongGame()
        self.math_ai = HandGestureAI(api_key)
        self.vir_keyboard = VirtualKeyboard()
        self.fit = ArmCurlsCounter()

        self.over = False
        self.show_options = False

        self.icon_img = cv2.imread(r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\MenuIcon2.png', cv2.IMREAD_UNCHANGED)
        self.icon_img = cv2.resize(self.icon_img, (40, 40))

    def loading(self):
        start_time = time.time()
        load_duration = 3  # Duration in seconds
        window_created = False

        while time.time() - start_time < load_duration:
            ret, img = self.cam.read()
            if not ret:
                continue
            img = cv2.flip(img, 1)

            progress = int((time.time() - start_time) / load_duration * 800) + 150
            angle = (time.time() - start_time) * 2 * np.pi  # Full rotation every second

            # Draw loading bar
            cv2.rectangle(img, (150, 300), (950, 400), (0, 255, 0), 5)
            cv2.rectangle(img, (150, 300), (progress, 400), (0, 255, 0), -1)
            cv2.rectangle(img, (progress, 295), (progress + 10, 405), (0, 255, 255), -1)

            # Draw rotating circle
            center = (550, 250)
            radius = 30
            circle_x = int(center[0] + radius * np.cos(angle))
            circle_y = int(center[1] + radius * np.sin(angle))
            cv2.circle(img, center, radius, (0, 0, 255), 2)
            cv2.circle(img, (circle_x, circle_y), 10, (255, 0, 0), -1)

            # Display loading text
            cv2.putText(img, "Loading...", (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Loading', img)
            cv2.waitKey(1)

            # Set flag when window is created
            if not window_created:
                window_created = True

        # Destroy the 'Loading' window only if it was created
        if window_created:
            cv2.destroyWindow('Loading')

    def draw_interface(self, img):
        buttons = [
            (100, 100, 200, 200, 'Volume Control', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\Volume+.png'),
            (300, 100, 400, 200, 'Paint App', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\PaintApp.png'),
            (500, 100, 600, 200, 'Math AI App', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\MathAI.png'),
            (700, 100, 800, 200, 'Presentation App', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\PresentationLogo.png'),
            (100, 300, 200, 400, 'Pong Game', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\PONGGame.png'),
            (300, 300, 400, 400, 'Virtual Mouse', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\VirtualMouse.png'),
            (500, 300, 600, 400, 'Fitness Tracker', r'C:\Users\bhush\OneDrive\Desktop\PAVAN\Projects\CV_Desktop\FitnessTracker.png')
        ]

        for (x1, y1, x2, y2, label, icon_path) in buttons:
            # Check if the icon file exists
            if os.path.exists(icon_path):
                # Load the icon image
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)

                if icon is not None:
                    # Resize the icon to fit the button
                    icon = cv2.resize(icon, (x2 - x1, y2 - y1))

                    # If the icon has an alpha channel (transparency)
                    if icon.shape[2] == 4:
                        # Create a mask from the alpha channel
                        mask = icon[:, :, 3]
                        # Remove the alpha channel from the icon
                        icon = icon[:, :, 0:3]

                        # Create a region of interest (ROI) on the main image
                        roi = img[y1:y2, x1:x2]

                        # Create a mask inverse
                        mask_inv = cv2.bitwise_not(mask)

                        # Black-out the area of icon in ROI
                        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                        # Take only region of icon from icon image
                        icon_fg = cv2.bitwise_and(icon, icon, mask=mask)

                        # Put icon in ROI and modify the main image
                        dst = cv2.add(img_bg, icon_fg)
                        img[y1:y2, x1:x2] = dst
                    else:
                        # If the icon doesn't have an alpha channel, simply copy it to the main image
                        img[y1:y2, x1:x2] = icon
                else:
                    print(f"Failed to load image: {icon_path}")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            else:
                print(f"Image file not found: {icon_path}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

            # Add text label below the icon
            lines = label.split()
            for i, line in enumerate(lines):
                cv2.putText(img, line, (x1, y2 + 30 + i * 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return img

    def draw_options(self, img):
        # cv2.rectangle(img, (900, 100), (1250, 400), (50, 50, 50), -1)  # Options background
        # cv2.putText(img, "Create File", (920, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(img, "Delete File", (920, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.putText(img, "Edit File", (920, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        overlay = img.copy()
        img = cv2.addWeighted(overlay, 0.3, np.zeros(img.shape, img.dtype), 0.7, 0)
        cv2.rectangle(overlay, (900, 100), (1250, 300), (50, 50, 50), -1)  # Options background
        cv2.putText(overlay, "Create File", (920, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "LogOut", (920, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Blend the overlay with the original image
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

        return img

    def create_file(self, cam):
        cv2.destroyAllWindows()
        self.vir_keyboard.add_content_to_file(cam)

    def overlay_image(self, background, overlay, x, y):
        """
        Overlays an image (overlay) onto another image (background) at the position (x, y).
        """

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

    def run(self, cam=None):
        if cam:
            self.cam = cam
        while True:
            ret, img = self.cam.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            hands, img = self.detector.findHands(img, flipType=False)

            if self.show_options:
                img = self.draw_options(img)

            if hands:

                lmlist = self.detector.findPosition(img)
                fingers = self.detector.fingersUp(hands[0])

                x1, y1 = lmlist[8][1], lmlist[8][2]
                x2, y2 = lmlist[12][1], lmlist[12][2]
                # if fingers == [1, 1, 1, 1, 1]:  exit condition
                #     self.over = True

                if fingers[1] and fingers[2]:
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 5)
                    if 1100 < x1 < 1120 and 50 < y1 < 70 or 1100 < x2 < 1120 and 50 < y2 < 70:
                        self.show_options = True
                elif fingers[1] and not fingers[2]:
                    cv2.circle(img, (x1, y1), 20, (0, 0, 255), -1)

                    if self.show_options:
                        if 900 < x1 < 1250 or 900 < x2 < 1250:
                            if 100 < y1 < 200:
                                self.create_file(self.cam)
                            elif 200 < y1 < 300:
                                cv2.destroyAllWindows()
                                break
                        else:
                            self.show_options = False

                    if 100 < x1 < 200 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.vol_control.run(self.cam)
                        self.loading()
                    elif 300 < x1 < 400 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.vir_paint.draw(self.cam)
                        self.loading()
                    elif 700 < x1 < 800 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.present_app.run(self.cam)
                        self.loading()
                    elif 100 < x1 < 200 and 300 < y1 < 400:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.pong_game.play_game(self.cam)
                        self.loading()
                    elif 300 < x1 < 400 and 300 < y1 < 400:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.vir_mouse.run(self.cam)
                        self.loading()
                    elif 500 < x1 < 600 and 300 < y1 < 400:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.fit.run(self.cam)
                        self.loading()
                    elif 500 < x1 < 600 and 100 < y1 < 200:
                        cv2.destroyAllWindows()
                        self.loading()
                        self.math_ai.run_app(self.cam)
                        self.loading()

            self.draw_interface(img)
            img = self.overlay_image(img, self.icon_img, 1080, 30)

            cv2.imshow('img', img)
            cv2.waitKey(1)
            if self.over:
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    app = CVApp()
    app.run()
