import face_recognition
import cv2
import numpy as np
import mysql.connector
from HandsGestureDetector import HandDetector as hd
import time
from Home_Page.home_page import CVApp
import threading
import queue
from Virtual_Keyboard.virtual_keyboard import VirtualKeyboard

class FaceRecognitionSystem:
    def __init__(self):
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="piyush@12345"
        )
        self.cursor = self.db.cursor()

        # Create database if not exists
        self.cursor.execute("CREATE DATABASE IF NOT EXISTS face_recognition_db")
        self.cursor.execute("USE face_recognition_db")

        # Create table if not exists
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                face_encoding BLOB
            )
        """)
        self.db.commit()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.app = CVApp()
        self.vir_Key = VirtualKeyboard()
        self.vir_Key.cam = self.cap

        self.countdown_start = 0
        self.is_counting_down = False
        self.capture_after_countdown = False
        self.action_after_capture = None

        self.detector = hd()
        self.text = ''
        self.flag = False
        self.ready_countdown_start = 0
        self.is_ready_counting_down = False

        self.face_encoding_queue = queue.Queue()
        self.face_encoding_thread = None
        self.is_encoding = False

    def draw_button(self, img, text, pos, size):
        cv2.rectangle(img, pos, (pos[0] + size[0], pos[1] + size[1]), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, text, (pos[0] + 10, pos[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def initiate_signup(self):
        _, frame = self.cap.read()
        name = self.vir_Key.enter_name(self.cap)
        if not name:
            self.text = "No name entered"
            return
        self.encode_face_async(frame)
        try:
            face_encoding = self.face_encoding_queue.get(timeout=5)
            if face_encoding is None:
                self.text = "No face detected"
                return
            result = self.signup(name, face_encoding)
            self.text = result
            if "added successfully" in result:
                self.text = f"User {name} added. Look at the camera for face confirmation."
                cv2.namedWindow('Face Recognition System', cv2.WINDOW_NORMAL)
                self.start_countdown('confirm')
            else:
                self.flag = False
        except queue.Empty:
            self.text = "Face encoding failed"
            self.flag = False

    def confirm_face(self, face_encoding):
        name = self.login(face_encoding)
        if "Welcome back" in name:
            self.text = f"{name} Face confirmed. Entering CV desktop in 3 seconds."
            self.flag = True
            self.start_ready_countdown()
        else:
            self.text = "Face confirmation failed. Please try again."
            self.flag = False
            self.start_countdown('confirm')

    def switch_to_keyboard_mode(self):
        cv2.destroyAllWindows()
        name = self.vir_Key.enter_name(self.cap)
        cv2.namedWindow('Face Recognition System', cv2.WINDOW_NORMAL)
        return name

    def encode_face_async(self, frame):
        if self.is_encoding:
            return
        self.is_encoding = True
        self.face_encoding_thread = threading.Thread(target=self._encode_face, args=(frame,))
        self.face_encoding_thread.start()

    def _encode_face(self, frame):
        if frame is None:
            self.face_encoding_queue.put(None)
            self.is_encoding = False
            return
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        if len(face_locations) == 0:
            self.face_encoding_queue.put(None)
            self.is_encoding = False
            return
        try:
            face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]
            self.face_encoding_queue.put(face_encoding)  # Store the numpy array directly
        except Exception as e:
            print(f"Error encoding face: {str(e)}")
            self.face_encoding_queue.put(None)
        self.is_encoding = False

    def start_countdown(self, action):
        self.countdown_start = time.time()
        self.is_counting_down = True
        self.capture_after_countdown = False
        self.action_after_capture = action

    def process_countdown(self, frame):
        if self.is_counting_down:
            elapsed_time = time.time() - self.countdown_start
            countdown_num = 3 - int(elapsed_time)
            if countdown_num >= 0:
                cv2.putText(frame, str(countdown_num), (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
                self.show_face_markings(frame)
                self.encode_face_async(frame)  # Continuously try to encode face during countdown
            else:
                self.is_counting_down = False
                self.capture_after_countdown = True

    def capture_and_process(self, frame):
        if self.capture_after_countdown:
            self.capture_after_countdown = False
            try:
                face_encoding = self.face_encoding_queue.get(timeout=1)
                if face_encoding is not None:
                    if self.action_after_capture == 'login':
                        self.text = self.login(face_encoding)
                    elif self.action_after_capture == 'confirm':
                        self.confirm_face(face_encoding)
                else:
                    self.text = "No face detected. Please try again."
                    self.start_countdown(self.action_after_capture)  # Restart countdown if face not detected
            except queue.Empty:
                self.text = "Face detection timed out. Please try again."
                self.start_countdown(self.action_after_capture)

    def signup(self, name, face_encoding):
        # Convert numpy array to bytes for storage
        face_encoding_bytes = face_encoding.tobytes()
        sql = "INSERT INTO users (name, face_encoding) VALUES (%s, %s)"
        val = (name, face_encoding_bytes)
        self.cursor.execute(sql, val)
        self.db.commit()
        return f'User {name} added successfully'

    def login(self, face_encoding):
        self.cursor.execute("SELECT * FROM users")
        users = self.cursor.fetchall()

        for user in users:
            # Convert stored bytes back to numpy array
            stored_encoding = np.frombuffer(user[2], dtype=np.float64)
            match = face_recognition.compare_faces([stored_encoding], face_encoding)[0]
            if match:
                self.flag = True
                return f"Welcome back, {user[1]}!"

        return "Face not recognized. Please sign up or try again."

    def show_face_markings(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    def start_ready_countdown(self):
        self.ready_countdown_start = time.time()
        self.is_ready_counting_down = True

    def process_ready_countdown(self, frame):
        if self.is_ready_counting_down:
            countdown_num = 3
            if countdown_num >= 0:
                cv2.putText(frame, f"Get ready to enter the CV desktop in {countdown_num}...",
                            (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                countdown_num -= 1
            else:
                self.is_ready_counting_down = False
                cv2.destroyAllWindows()
                self.app.run(self.cap)

    def run(self):
        login_button_pos = (100, 200)
        signup_button_pos = (400, 200)
        button_size = (200, 50)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if not self.is_counting_down:
                hands, img = self.detector.findHands(frame, flipType=False)

                if hands:
                    lmlist = self.detector.findPosition(img)
                    fingers = self.detector.fingersUp(hands[0])

                    if fingers[1] and not fingers[2]:
                        if 100 < lmlist[8][1] < 300 and 200 < lmlist[8][2] < 250:
                            self.start_countdown('login')
                        elif 400 < lmlist[8][1] < 600 and 200 < lmlist[8][2] < 250:
                            self.initiate_signup()

            self.process_countdown(frame)
            self.capture_and_process(frame)
            self.process_ready_countdown(frame)

            self.draw_button(frame, "Login", login_button_pos, button_size)
            self.draw_button(frame, "Signup", signup_button_pos, button_size)

            cv2.putText(frame, self.text, (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Face Recognition System', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if cv2.getWindowProperty('Face Recognition System', cv2.WND_PROP_VISIBLE) < 1:
                break

            if self.flag:
                time.sleep(1)  # Give a moment to see the final message
                cv2.destroyAllWindows()
                self.app.run(self.cap)
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()