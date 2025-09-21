import cv2
import os
from ultralytics import YOLO

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.model = YOLO("yolov8n-face.pt")  # đổi path nếu cần
        if not os.path.exists("faces"):
            os.makedirs("faces")

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Detect bằng YOLO
            results = self.model(frame, verbose=False)
            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    def save_face(self, user_id):
        ret, frame = self.cap.read()
        if ret:
            path = os.path.join("faces", f"{user_id}.jpg")
            cv2.imwrite(path, frame)
