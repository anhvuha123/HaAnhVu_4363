import os
import cv2
import json
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO
import face_recognition

# ==============================
# Khởi tạo thư mục và DB
# ==============================
os.makedirs("faces", exist_ok=True)
DB_FILE = "faces/faces_db.json"

if os.path.exists(DB_FILE):
    with open(DB_FILE, "r", encoding="utf-8") as f:
        faces_db = json.load(f)
else:
    faces_db = {}

# ==============================
# Load YOLO model
# ==============================
try:
    model = YOLO("yolov8n-face.pt")
    FACE_KEYPOINTS = True
except:
    st.warning("Không tìm thấy yolov8n-face.pt, dùng yolov8n.pt general")
    model = YOLO("yolov8n.pt")
    FACE_KEYPOINTS = False

# ==============================
# Session state
# ==============================
if "running" not in st.session_state:
    st.session_state["running"] = False
if "alert_message" not in st.session_state:
    st.session_state["alert_message"] = ""
if "student_encodings" not in st.session_state:
    st.session_state["student_encodings"] = []
if "prev_center" not in st.session_state:
    st.session_state["prev_center"] = None
if "frame_count" not in st.session_state:
    st.session_state["frame_count"] = 0

# ==============================
# Hàm giám sát camera lúc làm bài
# ==============================
def check_faces_test(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    results = model.predict(small_frame, conf=0.35)
    st.session_state["alert_message"] = ""

    if len(results[0].boxes) == 0:
        st.session_state["alert_message"] = "⚠️ Không thấy khuôn mặt!"
        return
    elif len(results[0].boxes) > 1:
        st.session_state["alert_message"] = "🚨 Có người khác xuất hiện!"
        return

    # 1 người
    box = results[0].boxes.xyxy[0].cpu().numpy()
    top, left, bottom, right = map(int, box[[1,0,3,2]] / 0.5)
    top, left = max(top,0), max(left,0)
    bottom, right = min(bottom,rgb_frame.shape[0]), min(right,rgb_frame.shape[1])
    face_rgb = rgb_frame[top:bottom, left:right]

    # Check face với ID học sinh
    st.session_state["frame_count"] += 1
    face_enc = None
    if st.session_state["frame_count"] % 5 == 0:
        try:
            if face_rgb.shape[0] < 150 or face_rgb.shape[1] < 150:
                face_small = cv2.resize(face_rgb, (150,150))
            else:
                face_small = face_rgb
            face_encs = face_recognition.face_encodings(face_small)
            if len(face_encs) > 0:
                face_enc = face_encs[0]
        except Exception as e:
            st.warning(f"Lỗi tính encoding: {e}")

    msg_face = ""
    if face_enc is not None and st.session_state["student_encodings"]:
        matches = face_recognition.compare_faces(st.session_state["student_encodings"], face_enc, tolerance=0.5)
        if not any(matches):
            msg_face = "🚨 Khuôn mặt lạ xuất hiện!"

    # Chuyển động
    center_x = (left + right)/2
    center_y = (top + bottom)/2
    msg_motion = ""
    if st.session_state["prev_center"] is not None:
        dx = abs(center_x - st.session_state["prev_center"][0])
        dy = abs(center_y - st.session_state["prev_center"][1])
        if dx > 40 or dy > 40:
            msg_motion = "⚠️ Chuyển động lớn, tập trung!"
    st.session_state["prev_center"] = (center_x, center_y)

    if msg_face:
        st.session_state["alert_message"] = msg_face
    elif msg_motion:
        st.session_state["alert_message"] = msg_motion
    else:
        st.session_state["alert_message"] = "✅ Khuôn mặt hợp lệ."

# ==============================
# Streamlit GUI
# ==============================
st.set_page_config(page_title="Face Detection & Test", layout="wide")
st.title("📷 Quản lý Khuôn Mặt & Bài Test Online")

menu = st.sidebar.radio("Chức năng", [
    "Quét & Lưu Khuôn Mặt",
    "Xem DS Khuôn Mặt",
    "Xóa Khuôn Mặt",
    "Làm Bài Test Online"
])

# ==============================
# 1. Quét & Lưu
# ==============================
if menu == "Quét & Lưu Khuôn Mặt":
    st.header("➕ Quét & Lưu Khuôn Mặt")
    face_id = st.text_input("Nhập ID học sinh:")
    start = st.button("📸 Bắt đầu quét")
    if start and face_id:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        saved_encodings = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame)
            annotated = results[0].plot()

            if len(results[0].boxes) > 0:
                box = results[0].boxes.xyxy[0].cpu().numpy()
                top, left, bottom, right = map(int, box[[1,0,3,2]])
                face_rgb = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                face_encs = face_recognition.face_encodings(face_rgb)
                if face_encs:
                    saved_encodings.append(face_encs[0])
                filename = f"faces/face_{face_id}.jpg"
                cv2.imwrite(filename, frame[top:bottom, left:right])
                faces_db[face_id] = filename
                with open(DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(faces_db, f, indent=4, ensure_ascii=False)

            stframe.image(annotated, channels="BGR")
            if len(saved_encodings) >= 5:
                st.session_state["student_encodings"] = saved_encodings
                break
        cap.release()
        st.success(f"✅ Đã lưu khuôn mặt ID {face_id}")

# ==============================
# 2. Xem DS
# ==============================
elif menu == "Xem DS Khuôn Mặt":
    st.header("📂 Danh Sách Khuôn Mặt")
    if not faces_db:
        st.warning("Chưa có khuôn mặt nào!")
    else:
        for fid, path in faces_db.items():
            st.image(path, caption=f"ID: {fid}", width=150)

# ==============================
# 3. Xóa
# ==============================
elif menu == "Xóa Khuôn Mặt":
    st.header("🗑 Xóa Khuôn Mặt")
    del_id = st.text_input("Nhập ID muốn xóa:")
    if st.button("Xóa"):
        if del_id in faces_db:
            if os.path.exists(faces_db[del_id]):
                os.remove(faces_db[del_id])
            del faces_db[del_id]
            with open(DB_FILE, "w", encoding="utf-8") as f:
                json.dump(faces_db, f, indent=4, ensure_ascii=False)
            st.success(f"✅ Đã xóa ID {del_id}")
        else:
            st.error("❌ ID không tồn tại!")

# ==============================
# 4. Làm Bài Test Online
# ==============================
elif menu == "Làm Bài Test Online":
    st.header("📝 Bài Test Online Có Giám Sát Camera")
    test_id = st.text_input("Nhập ID học sinh:")

    if st.button("▶ Bắt đầu giám sát"):
        if test_id in faces_db:
            known_image = face_recognition.load_image_file(faces_db[test_id])
            known_encodings = face_recognition.face_encodings(known_image)
            if known_encodings:
                st.session_state["student_encodings"] = known_encodings
                st.session_state["running"] = True
                st.success("✅ Giám sát bắt đầu. Hãy làm bài test!")

                stframe = st.empty()
                alert_placeholder = st.empty()
                cap = cv2.VideoCapture(0)

                while st.session_state["running"]:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    check_faces_test(frame)

                    msg = st.session_state.get("alert_message","")
                    if msg:
                        if "lạ xuất hiện" in msg or "Không thấy" in msg:
                            alert_placeholder.error(msg)
                        elif "⚠️" in msg:
                            alert_placeholder.warning(msg)
                        else:
                            alert_placeholder.success(msg)
                    time.sleep(0.15)
                cap.release()

        else:
            st.error("❌ ID không tồn tại trong DB!")
    if st.button("■ Dừng giám sát"):
        st.session_state["running"] = False
        st.session_state["alert_message"] = ""
        st.session_state["prev_center"] = None
        st.session_state["frame_count"] = 0
        st.success("■ Giám sát đã dừng.")
  
  


