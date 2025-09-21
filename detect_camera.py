# detect_camera.py
import cv2

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 là camera mặc định
        if not self.cap.isOpened():
            raise Exception("Không mở được camera")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def save_face(self, user_id):
        ret, frame = self.cap.read()
        if not ret:
            return False
        cv2.imwrite(f"faces/{user_id}.jpg", frame)
        return True