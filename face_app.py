import os
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from ultralytics import YOLO
import threading
import subprocess
import platform
import face_recognition
import numpy as np
import json


# Khởi tạo

os.makedirs("faces", exist_ok=True)
DB_FILE = "faces/faces_db.json"

# Load DB (id -> filename)
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r", encoding="utf-8") as f:
        faces_db = json.load(f)
else:
    faces_db = {}

# Load YOLO model (nên thay bằng model face riêng: yolov8n-face.pt)
model = YOLO("yolov8n.pt")


# Quét & lưu khuôn mặt

def scan_faces():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            face = frame[top:bottom, left:right]
            if face.size > 0:
                # Hỏi ID
                face_id = simpledialog.askstring("Nhập ID", "Nhập ID cho khuôn mặt này:")
                if not face_id:
                    continue
                filename = f"faces/face_{face_id}.jpg"
                cv2.imwrite(filename, face)

                # Lưu vào DB
                faces_db[face_id] = filename
                with open(DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(faces_db, f, indent=4, ensure_ascii=False)

                messagebox.showinfo("Thành công", f"Đã lưu khuôn mặt ID {face_id}")

        cv2.imshow("Face Scanner (ấn Q để thoát)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Mở thư mục faces

def open_faces_folder():
    path = os.path.abspath("faces")
    if platform.system() == "Windows":
        subprocess.Popen(f'explorer "{path}"')
    elif platform.system() == "Darwin":  # macOS
        subprocess.Popen(["open", path])
    else:  # Linux
        subprocess.Popen(["xdg-open", path])


# Test face theo ID

def test_face():
    test_id = simpledialog.askstring("Nhập ID", "Nhập ID khuôn mặt muốn kiểm tra:")
    if not test_id or test_id not in faces_db:
        messagebox.showerror("Lỗi", f"ID {test_id} chưa có trong dữ liệu!")
        return

    face_file = faces_db[test_id]
    known_image = face_recognition.load_image_file(face_file)
    known_encoding = face_recognition.face_encodings(known_image)
    if len(known_encoding) == 0:
        messagebox.showerror("Lỗi", f"Không phát hiện khuôn mặt trong {face_file}")
        return
    known_encoding = known_encoding[0]

    cap = cv2.VideoCapture(0)
    matched = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame)
        for encoding in encodings:
            match = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.5)
            if match[0]:
                matched = True
                break

        cv2.imshow("Face Test (ấn Q để thoát)", annotated)
        if matched:
            messagebox.showinfo("Kết quả", f"Khuôn mặt khớp với ID {test_id}")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if not matched:
        messagebox.showerror("Kết quả", f"Khuôn mặt KHÔNG khớp với ID {test_id}")

    cap.release()
    cv2.destroyAllWindows()


# Xóa khuôn mặt theo ID

def delete_face():
    del_id = simpledialog.askstring("Nhập ID", "Nhập ID khuôn mặt muốn xóa:")
    if not del_id or del_id not in faces_db:
        messagebox.showerror("Lỗi", f"ID {del_id} không tồn tại!")
        return

    face_file = faces_db[del_id]
    if os.path.exists(face_file):
        os.remove(face_file)

    del faces_db[del_id]
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(faces_db, f, indent=4, ensure_ascii=False)

    messagebox.showinfo("Xong", f"Đã xóa khuôn mặt ID {del_id}")


# GUI Tkinter

root = tk.Tk()
root.title("Face Detection App")
root.geometry("400x300")

btn1 = tk.Button(root, text="1. Quét & Lưu Khuôn Mặt", command=lambda: threading.Thread(target=scan_faces).start(), height=2, width=40)
btn1.pack(pady=10)

btn2 = tk.Button(root, text="2. Xem Khuôn Mặt Đã Lưu", command=open_faces_folder, height=2, width=40)
btn2.pack(pady=10)

btn3 = tk.Button(root, text="3. Test Khuôn Mặt Theo ID", command=lambda: threading.Thread(target=test_face).start(), height=2, width=40)
btn3.pack(pady=10)

btn4 = tk.Button(root, text="4. Xóa Khuôn Mặt Theo ID", command=delete_face, height=2, width=40)
btn4.pack(pady=10)

root.mainloop()
