import cv2
import face_recognition
import numpy as np
import sqlite3
import json

# ==============================================================================
# CÀI ĐẶT TỐI ƯU CHO MÁY CẤU HÌNH THẤP
# ------------------------------------------------------------------------------
# 1. Tăng SKIP_FRAMES để giảm số lần xử lý. Ví dụ: 10 nghĩa là chỉ xử lý
#    1 khung hình trong mỗi 10 khung hình nhận được từ webcam.
#    => Tăng giá trị này nếu máy vẫn chậm.
SKIP_FRAMES = 10

# 2. Giảm RESIZE_FACTOR để làm ảnh nhỏ hơn, giúp xử lý nhanh hơn.
#    => Giảm giá trị này nếu máy vẫn chậm, nhưng sẽ khó nhận diện mặt ở xa.
RESIZE_FACTOR = 0.9

# 3. Ngưỡng nhận diện. Giữ ở mức 0.6 là hợp lý cho máy yếu.
RECOGNITION_TOLERANCE = 0.6
# ==============================================================================

DB_NAME = 'student_faces.db'

# (Các hàm CSDL: create_table, add_student_to_db, v.v... giữ nguyên như cũ)
def create_table():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                class TEXT NOT NULL,
                face_encoding TEXT NOT NULL UNIQUE
            )
        ''')
    print(f"Đã kiểm tra/tạo bảng 'students' trong {DB_NAME}.")

def add_student_to_db(name, student_class, face_encoding):
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            face_encoding_json = json.dumps(face_encoding.tolist())
            cursor.execute("INSERT INTO students (name, class, face_encoding) VALUES (?, ?, ?)",
                           (name, student_class, face_encoding_json))
        print(f"Thành công: Đã thêm học sinh: {name} - Lớp: {student_class} vào CSDL.")
        return True
    except sqlite3.IntegrityError:
        print(f"Lỗi: Khuôn mặt của '{name}' có thể đã tồn tại trong CSDL.")
        return False

def get_all_students_from_db():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, class, face_encoding FROM students")
            rows = cursor.fetchall()
        students = []
        for row in rows:
            student_id, name, student_class, face_encoding_json = row
            face_encoding = np.array(json.loads(face_encoding_json))
            students.append({"id": student_id, "name": name, "class": student_class, "encoding": face_encoding})
        return students
    except sqlite3.Error as e:
        print(f"Lỗi truy cập CSDL: {e}")
        return []

def delete_student_from_db(student_id):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    print(f"Đã xóa học sinh với ID: {student_id} khỏi CSDL.")

# --- Logic Nhận diện Khuôn mặt Tối ưu ---
def run_face_recognition():
    create_table()
    
    known_students = get_all_students_from_db()
    known_face_encodings = [s["encoding"] for s in known_students]
    known_face_data = [(s["id"], s["name"], s["class"]) for s in known_students]

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Lỗi: Không thể truy cập webcam.")
        return

    print("\n--- CHƯƠNG TRÌNH NHẬN DIỆN (PHIÊN BẢN CHO MÁY YẾU) ---")
    print(f"Chế độ: Xử lý 1/{SKIP_FRAMES} khung hình.")
    print("  'a' -> Thêm 'Người lạ' | 'd' -> Xóa | 'l' -> Liệt kê | 'q' -> Thoát")
    print("---------------------------------------------------------")

    frame_count = 0
    face_locations = []
    face_names = []
    last_detected_student_id = None
    last_detected_unknown_encoding = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Hiển thị luôn để video không bị đứng hình
        display_frame = frame.copy()
        frame_count += 1

        # Chỉ xử lý nhận diện mỗi SKIP_FRAMES lần
        if frame_count % SKIP_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Tìm tất cả khuôn mặt trong khung hình nhỏ
            all_face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            all_face_encodings = face_recognition.face_encodings(rgb_small_frame, all_face_locations)

            # Nếu có khuôn mặt, chỉ lấy khuôn mặt lớn nhất
            if all_face_locations:
                areas = [(b-t)*(r-l) for (t, r, b, l) in all_face_locations]
                max_idx = np.argmax(areas)
                face_locations = [all_face_locations[max_idx]]
                face_encodings = [all_face_encodings[max_idx]]
            else:
                face_locations = []
                face_encodings = []

            face_names = []
            last_detected_student_id = None
            last_detected_unknown_encoding = None

            for face_encoding in face_encodings:
                name, student_class, student_id = "Người lạ", "", None
                if known_face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RECOGNITION_TOLERANCE)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        student_id, name, student_class = known_face_data[best_match_index]
                        last_detected_student_id = student_id

                if student_id is None:
                    last_detected_unknown_encoding = face_encoding
                face_names.append(f"{name} {student_class}".strip())

        # Hiển thị box (dù frame không nhận diện, vẫn vẽ lại box cũ để video không đứng hình)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top = int(top / RESIZE_FACTOR)
            right = int(right / RESIZE_FACTOR)
            bottom = int(bottom / RESIZE_FACTOR)
            left = int(left / RESIZE_FACTOR)

            color = (0, 0, 255) if "Người lạ" in name else (0, 255, 0)
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(display_frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        cv2.imshow('He thong Nhan dien Khuon mat', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # Các hành động thêm/xóa/liệt kê vẫn giữ nguyên
        elif key == ord('a'):
            # ... (giữ nguyên logic phím 'a')
            if last_detected_unknown_encoding is not None:
                print("\n--- THÊM HỌC SINH MỚI ---")
                name = input(">> Nhập tên học sinh: ")
                student_class = input(f">> Nhập lớp của {name}: ")
                if name and student_class:
                    if add_student_to_db(name, student_class, last_detected_unknown_encoding):
                        known_students = get_all_students_from_db()
                        known_face_encodings = [s["encoding"] for s in known_students]
                        known_face_data = [(s["id"], s["name"], s["class"]) for s in known_students]
                else:
                    print("Tên hoặc lớp không được để trống. Hủy thêm.")
            else:
                print("\n[!] Không có 'Người lạ' trong khung hình để thêm.")
        elif key == ord('d'):
            # ... (giữ nguyên logic phím 'd')
            if last_detected_student_id:
                print(f"\n--- XÓA HỌC SINH (ID: {last_detected_student_id}) ---")
                confirm = input(f">> Bạn có chắc muốn xóa học sinh có ID {last_detected_student_id}? (y/n): ").lower()
                if confirm == 'y':
                    delete_student_from_db(last_detected_student_id)
                    known_students = get_all_students_from_db()
                    known_face_encodings = [s["encoding"] for s in known_students]
                    known_face_data = [(s["id"], s["name"], s["class"]) for s in known_students]
                    last_detected_student_id = None
                else:
                    print("Đã hủy thao tác xóa.")
            else:
                print("\n[!] Không có học sinh nào (đã được nhận diện) để xóa.")
        elif key == ord('l'):
            # ... (giữ nguyên logic phím 'l')
            print("\n--- DANH SÁCH HỌC SINH TRONG CSDL ---")
            students_list = get_all_students_from_db()
            if not students_list: print("Chưa có học sinh nào.")
            else:
                for s in students_list: print(f"  - ID: {s['id']}, Tên: {s['name']}, Lớp: {s['class']}")
            print("--------------------------------------")

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recognition()