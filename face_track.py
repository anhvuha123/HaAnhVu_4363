import os
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO

# Tạo thư mục lưu khuôn mặt nếu chưa có
os.makedirs("faces", exist_ok=True)

# Load model YOLOv8 đã train (thay yolov8n.pt bằng best.pt nếu bạn có model face)
model = YOLO("yolov8n.pt")

# Mở webcam
cap = cv2.VideoCapture(0)

# List để lưu các đặc trưng khuôn mặt đã nhận diện
known_faces = []
known_face_ids = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking + giữ ID
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Vẽ bounding box + ID lên frame
    annotated = results[0].plot()

    # ==== 📌 CẮT VÀ LƯU KHUÔN MẶT THEO ID ====
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()   # [x1,y1,x2,y2]
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else []
        for box, face_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            if face.size > 0:  # tránh lỗi khi crop ngoài khung
                # Chuyển đổi khuôn mặt thành đặc trưng (face encoding)
                face_encoding = face_recognition.face_encodings(face)
                
                if face_encoding:
                    face_encoding = face_encoding[0]
                    
                    # Kiểm tra nếu khuôn mặt này đã có trong danh sách
                    matches = face_recognition.compare_faces(known_faces, face_encoding)
                    
                    if True in matches:
                        # Nếu khuôn mặt trùng khớp, cập nhật ID
                        first_match_index = matches.index(True)
                        face_id = known_face_ids[first_match_index]
                    else:
                        # Nếu không trùng khớp, thêm vào danh sách
                        known_faces.append(face_encoding)
                        known_face_ids.append(int(face_id))
                
                # Lưu khuôn mặt với ID mới (nếu cần)
                filename = f"faces/face_{int(face_id)}.jpg"
                cv2.imwrite(filename, face)

    # Hiển thị webcam
    cv2.imshow("YOLOv8 Face Tracking", annotated)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Lưu database khuôn mặt
DB_FILE = "faces/faces_db.json"