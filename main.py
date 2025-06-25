# main_app.py
"""
Ứng dụng chính:
- Điều phối các module khác.
- Khởi tạo webcam và vòng lặp chính.
- Xử lý giao diện người dùng (vẽ lên màn hình, nhận phím bấm).
"""
import cv2
import config
import database_manager as db
import face_processor

def main():
    # --- KHỞI TẠO ---
    db.create_table()
    known_students = db.get_all_students()

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Lỗi: Không thể truy cập webcam.")
        return

    print("\n--- CHƯƠG TRÌNH NHẬN DIỆN (PHIÊN BẢN MODULE) ---")
    print(f"Chế độ: Xử lý 1/{config.SKIP_FRAMES} khung hình.")
    print("  'a' -> Thêm 'Người lạ' | 's' -> Sửa | 'd' -> Xóa ")
    print("  'l' -> Liệt kê        | 'q' -> Thoát")
    print("---------------------------------------------------------")

    frame_count = 0
    face_locations = []
    face_names = []
    last_detected_student_id = None
    last_unknown_encoding = None

    # --- VÒNG LẶP CHÍNH ---
    while True:
        ret, frame = video_capture.read()
        if not ret: break

        if frame_count % config.SKIP_FRAMES == 0:
            # Module 1: Tiền xử lý và phát hiện
            temp_locations, temp_encodings = face_processor.process_frame_for_faces(frame)

            if temp_encodings:
                face_locations = temp_locations # Cập nhật vị trí để vẽ
                # Module 3: Truy xuất thông tin
                face_names, identified_ids, last_unknown_encoding = face_processor.identify_faces(temp_encodings, known_students)
                
                # Tìm ID cuối cùng được nhận diện (không phải người lạ)
                valid_ids = [id for id in identified_ids if id is not None]
                last_detected_student_id = valid_ids[-1] if valid_ids else None
            else:
                # Nếu không có khuôn mặt nào, xóa các box cũ
                face_locations, face_names = [], []

        frame_count += 1

        # Giao diện: Vẽ kết quả lên màn hình
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top = int(top / config.RESIZE_FACTOR)
            right = int(right / config.RESIZE_FACTOR)
            bottom = int(bottom / config.RESIZE_FACTOR)
            left = int(left / config.RESIZE_FACTOR)

            color = (0, 0, 255) if "Người lạ" in name else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('He thong Nhan dien Khuon mat', frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Module 2: Các thao tác CSDL dựa trên phím bấm
        if key == ord('a'):
            if last_unknown_encoding is not None:
                print("\n--- THÊM HỌC SINH MỚI ---")
                name = input(">> Nhập tên: ")
                s_class = input(f">> Nhập lớp của {name}: ")
                if name and s_class and db.add_student(name, s_class, last_unknown_encoding):
                    print("[App] Đang tải lại danh sách học sinh...")
                    known_students = db.get_all_students()
            else:
                print("\n[!] Không có 'Người lạ' trong khung hình để thêm.")
        
        elif key == ord('d'):
            if last_detected_student_id:
                confirm = input(f">> Chắc chắn xóa học sinh ID {last_detected_student_id}? (y/n): ").lower()
                if confirm == 'y' and db.delete_student(last_detected_student_id):
                    print(f"[App] Đã xóa. Đang tải lại danh sách...")
                    known_students = db.get_all_students()
                    face_locations, face_names = [], [] # Xóa box trên màn hình
            else:
                print("\n[!] Không có học sinh nào (đã nhận diện) để xóa.")

        elif key == ord('s'):
            id_to_edit = input(">> Nhập ID học sinh cần sửa: ")
            if id_to_edit.isdigit():
                new_name = input(f">> Nhập tên mới cho ID {id_to_edit}: ")
                new_class = input(f">> Nhập lớp mới cho ID {id_to_edit}: ")
                if new_name and new_class and db.update_student(int(id_to_edit), new_name, new_class):
                    print(f"[App] Đã cập nhật. Đang tải lại danh sách...")
                    known_students = db.get_all_students()
                else:
                    print("[!] Cập nhật thất bại hoặc không có thay đổi.")
            else:
                print("[!] ID không hợp lệ.")

        elif key == ord('l'):
            print("\n--- DANH SÁCH HỌC SINH ---")
            all_students = db.get_all_students()
            if not all_students: print("CSDL trống.")
            else:
                for s in all_students: print(f"  - ID: {s['id']}, Tên: {s['name']}, Lớp: {s['class']}")
            print("---------------------------")

    # --- DỌN DẸP ---
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()