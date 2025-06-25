# face_processor.py
"""
Module 1 & 3: Tiền xử lý ảnh, phát hiện và nhận diện khuôn mặt.
- Tiền xử lý: Thay đổi kích thước, chuyển đổi màu sắc.
- Nhận diện: Tìm vị trí khuôn mặt và tạo mã hóa (encoding).
- Truy xuất thông tin: So sánh mã hóa mới với danh sách đã biết để tìm ra người quen.
"""
import cv2
import face_recognition
import numpy as np
from config import RESIZE_FACTOR, RECOGNITION_TOLERANCE

def process_frame_for_faces(frame):
    """
    Tiền xử lý một khung hình và phát hiện các khuôn mặt.
    Trả về: vị trí các khuôn mặt và mã hóa của chúng.
    """
    # Thay đổi kích thước và màu sắc để xử lý nhanh hơn
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Phát hiện vị trí và mã hóa khuôn mặt
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    return face_locations, face_encodings

def identify_faces(face_encodings, known_students):
    """
    Truy xuất thông tin của các khuôn mặt được nhận diện.
    So sánh các khuôn mặt tìm thấy với CSDL.
    """
    if not known_students:
        # Nếu CSDL trống, tất cả đều là người lạ
        return ["Người lạ" for _ in face_encodings], [None for _ in face_encodings], [None for _ in face_encodings]

    known_face_encodings = [s["encoding"] for s in known_students]
    known_face_data = [(s["id"], s["name"], s["class"]) for s in known_students]

    identified_names = []
    identified_ids = []
    unidentified_encodings = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=RECOGNITION_TOLERANCE)
        
        name, student_class, student_id = "Người lạ", "", None

        # Tìm người khớp nhất
        face_distances = face_recognition.face_distance(known_face_encodings, encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            student_id, name, student_class = known_face_data[best_match_index]
        
        identified_ids.append(student_id)
        identified_names.append(f"{name} {student_class}".strip())

        # Nếu là người lạ, lưu lại encoding để có thể thêm vào CSDL
        if student_id is None:
            unidentified_encodings.append(encoding)

    # Chỉ lấy encoding của người lạ đầu tiên được phát hiện
    last_unknown_encoding = unidentified_encodings[0] if unidentified_encodings else None
    
    return identified_names, identified_ids, last_unknown_encoding