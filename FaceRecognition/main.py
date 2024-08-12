import face_recognition
import cv2
import numpy as np
import mysql.connector
from mysql.connector import pooling
from HandsGestureDetector import HandDetector as hd
import time
from Home_Page.home_page import CVApp
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional
from Virtual_Keyboard.virtual_keyboard import VirtualKeyboard
import os


class DatabaseManager:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', 'piyush@12345'),
            'database': 'face_recognition_db'
        }
        self.init_database()
        self.create_connection_pool()

    def init_database(self):
        try:
            conn = mysql.connector.connect(
                host=self.db_config['host'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']}")
            cursor.execute(f"USE {self.db_config['database']}")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    face_encoding LONGBLOB
                )
            """)
            conn.commit()
        except mysql.connector.Error as err:
            print(f"Error initializing database: {err}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def create_connection_pool(self):
        try:
            self.pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=5,
                **self.db_config
            )
        except mysql.connector.Error as err:
            print(f"Error creating connection pool: {err}")

    def get_connection(self):
        return self.pool.get_connection()

    def insert_user(self, name, face_encoding):
        query = "INSERT INTO users (name, face_encoding) VALUES (%s, %s)"
        face_encoding_bytes = face_encoding.tobytes() if face_encoding is not None else None
        with self.get_connection() as conn:
            with conn.cursor(prepared=True) as cursor:
                try:
                    cursor.execute(query, (name, face_encoding_bytes))
                    conn.commit()
                    return True
                except mysql.connector.Error as err:
                    print(f"Error inserting user: {err}")
                    conn.rollback()
                    return False

    def get_all_users(self):
        query = "SELECT * FROM users"
        with self.get_connection() as conn:
            with conn.cursor(prepared=True) as cursor:
                try:
                    cursor.execute(query)
                    return cursor.fetchall()
                except mysql.connector.Error as err:
                    print(f"Error fetching users: {err}")
                    return None

    def login_user(self, face_encoding):
        users = self.get_all_users()
        if not users:
            print('Didnt got any user from the database')
            return None

        for user in users:
            if user[2] is not None:  # Check if face_encoding is not None
                stored_encoding = np.frombuffer(user[2], dtype=np.float64)
                # Use a lower tolerance for stricter matching
                if face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.5)[0]:
                    return user
        print('Didnt match any user from the database')
        return None


class Button:
    def __init__(self, text, pos, size):
        self.text = text
        self.pos = pos
        self.size = size

    def draw(self, frame, button_color, text_color):
        cv2.rectangle(frame, self.pos, (self.pos[0] + self.size[0], self.pos[1] + self.size[1]), button_color,
                      cv2.FILLED)
        cv2.putText(frame, self.text, (self.pos[0] + 20, self.pos[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    def is_over(self, x, y):
        return self.pos[0] < x < self.pos[0] + self.size[0] and self.pos[1] < y < self.pos[1] + self.size[1]


class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = np.random.randn() * 2
        self.vy = np.random.randn() * 2
        self.radius = np.random.randint(3, 7)
        self.color = tuple(np.random.randint(0, 255, 3).tolist())
        self.life = np.random.randint(20, 40)
        self.alive = True

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        if self.life <= 0:
            self.alive = False

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)


class FaceRecognitionSystem:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.app = CVApp()
        self.vir_Key = VirtualKeyboard()
        self.vir_Key.cam = self.cap

        self.detector = hd()
        self.text = ''
        self.countdown_start = 0
        self.is_counting_down = False
        self.action_after_countdown = None

        self.face_encoding_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.is_encoding = False

        self.max_encoding_attempts = 5
        self.current_encoding_attempt = 0

        self.bg_color = (245, 230, 200)  # Light beige background
        self.button_color = (70, 150, 180)  # Teal buttons
        self.text_color = (50, 50, 50)  # Dark gray text
        self.highlight_color = (255, 170, 50)  # Orange highlight

        # UI elements
        self.login_button = Button("Login", (100, 300), (300, 80))
        self.signup_button = Button("Signup", (500, 300), (300, 80))
        self.close_button = Button("Close", (1050, 50), (110, 80))
        self.particles = []

    def draw_ui(self, frame):
        title = "Face Recognition System"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        title_x = (frame.shape[1] - title_size[0]) // 2
        cv2.rectangle(frame, (title_x - 10, 70), (title_x + title_size[0] + 10, 110), (0, 0, 0), -1)
        cv2.putText(frame, title, (title_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        # Draw buttons
        self.login_button.draw(frame, self.button_color, self.text_color)
        self.signup_button.draw(frame, self.button_color, self.text_color)
        self.close_button.draw(frame, (200, 50, 50), (255, 255, 255))  # Red button with white text

        # Draw particles
        for particle in self.particles:
            particle.update()
            particle.draw(frame)

        # Remove dead particles
        self.particles = [p for p in self.particles if p.alive]

        # Draw status text on a black rectangle
        if self.text:
            text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (50, 500), (60 + text_size[0], 530 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(frame, self.text, (55, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def handle_interaction(self, x, y):
        if self.login_button.is_over(x, y):
            self.start_countdown('login')
            self.create_particles(x, y)
        elif self.signup_button.is_over(x, y):
            self.initiate_signup()
            self.create_particles(x, y)
        elif self.close_button.is_over(x, y):
            return True  # Signal to terminate the process
        return False

    def create_particles(self, x, y):
        for _ in range(20):
            self.particles.append(Particle(x, y))

    def draw_button(self, img: np.ndarray, text: str, pos: Tuple[int, int], size: Tuple[int, int]) -> None:
        cv2.rectangle(img, pos, (pos[0] + size[0], pos[1] + size[1]), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, text, (pos[0] + 10, pos[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def initiate_signup(self) -> None:
        name = self.vir_Key.enter_name(self.cap)
        if not name:
            self.text = "No name entered"
            return
        self.signup_name = name
        self.text = f"User {name} added. Look at the camera for face capture."
        cv2.namedWindow('Face Recognition System', cv2.WINDOW_NORMAL)
        self.start_countdown('signup')

    def encode_face_async(self, frame: np.ndarray) -> None:
        if not self.is_encoding:
            self.is_encoding = True
            self.thread_pool.submit(self._encode_face, frame)

    def _encode_face(self, frame: np.ndarray) -> None:
        if frame is None:
            self.face_encoding_queue.put(None)
            self.is_encoding = False
            return
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        if not face_locations:
            self.face_encoding_queue.put(None)
            self.is_encoding = False
            return
        try:
            face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)[0]
            self.face_encoding_queue.put(face_encoding)
        except Exception as e:
            print(f"Error encoding face: {str(e)}")
            self.face_encoding_queue.put(None)
        self.is_encoding = False

    def signup(self, name, face_encoding):
        if self.db_manager.insert_user(name, face_encoding):
            return f'User {name} added successfully'
        return f'Failed to add user {name}'

    def login(self, face_encoding):
        user = self.db_manager.login_user(face_encoding)
        if user:
            return f"Welcome back, {user[1]}!"
        return "Face not recognized. Please sign up to continue."

    def start_countdown(self, action: str) -> None:
        self.countdown_start = time.time()
        self.is_counting_down = True
        self.action_after_countdown = action
        self.current_encoding_attempt = 0

    def process_countdown(self, frame: np.ndarray) -> None:
        if self.is_counting_down:
            elapsed_time = time.time() - self.countdown_start
            countdown_num = 3 - int(elapsed_time)
            if countdown_num >= 0:
                cv2.putText(frame, str(countdown_num), (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
                self.encode_face_async(frame)
            else:
                self.is_counting_down = False
                if self.action_after_countdown == 'signup':
                    self.complete_signup()
                elif self.action_after_countdown == 'login':
                    self.complete_login()

    def complete_signup(self) -> None:
        face_encoding = self.get_face_encoding()
        if face_encoding is not None:
            result = self.signup(self.signup_name, face_encoding)
            self.text = f"{result}. Get ready to enter CV desktop in 3 seconds."
            self.start_countdown('enter_cv_desktop')

    def complete_login(self) -> None:
        face_encoding = self.get_face_encoding()
        if face_encoding is not None:
            login_result = self.login(face_encoding)
            if "Welcome back" in login_result:
                self.text = f"{login_result} Get ready to enter CV desktop in 3 seconds."
                self.start_countdown('enter_cv_desktop')
            else:
                self.text = login_result
        else:
            self.text = "Face not detected. Please try again."

    def get_face_encoding(self) -> Optional[np.ndarray]:
        while self.current_encoding_attempt < self.max_encoding_attempts:
            try:
                face_encoding = self.face_encoding_queue.get(timeout=1)
                if face_encoding is not None:
                    return face_encoding
                # self.current_encoding_attempt += 1
            except queue.Empty:
                self.current_encoding_attempt += 1

        return None

    def reset_system(self):
        self.text = ''
        self.countdown_start = 0
        self.is_counting_down = False
        self.action_after_countdown = None

        self.face_encoding_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.is_encoding = False

        self.max_encoding_attempts = 5
        self.current_encoding_attempt = 0

        self.button_color = (70, 150, 180)  # Teal buttons
        self.text_color = (255, 255, 255)  # Dark gray text
        self.highlight_color = (255, 170, 50)  # Orange highlight

        # UI elements
        self.login_button = Button("Login", (100, 300), (300, 80))
        self.signup_button = Button("Signup", (500, 300), (300, 80))
        self.particles = []

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = self.draw_ui(frame)

            if not self.is_counting_down:
                hands, _ = self.detector.findHands(frame, flipType=False)
                if hands:
                    lmlist = self.detector.findPosition(frame)
                    fingers = self.detector.fingersUp(hands[0])
                    if fingers[1] and not fingers[2]:
                        if self.handle_interaction(lmlist[8][1], lmlist[8][2]):
                            break  # Terminate the process if close button is clicked

            self.process_countdown(frame)

            cv2.imshow('Face Recognition System', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Recognition System',
                                                                          cv2.WND_PROP_VISIBLE) < 1:
                break

            if self.action_after_countdown == 'enter_cv_desktop' and not self.is_counting_down:
                time.sleep(1)
                cv2.destroyAllWindows()
                self.app.run(self.cap)
                self.action_after_countdown = None
                self.reset_system()

        self.cap.release()
        cv2.destroyAllWindows()
        self.thread_pool.shutdown()


if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()