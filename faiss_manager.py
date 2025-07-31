import faiss
import numpy as np
import os
import json
import database_manager as db

# Đặt tên file cho chỉ mục và file ánh xạ ID
FAISS_INDEX_FILE = "student_faces.index"
ID_MAPPING_FILE = "student_ids.json"

def build_and_save_index():
    """
    Lấy tất cả các face encoding từ CSDL, xây dựng chỉ mục Faiss và lưu ra file.
    """
    print("[Faiss] Bắt đầu xây dựng chỉ mục từ CSDL...")
    students = db.get_all_students()

    if not students:
        print("[Faiss] CSDL rỗng, không có gì để xây dựng.")
        return

    # Lấy ra các mã hóa khuôn mặt và ID tương ứng
    face_encodings = [s["face_encoding"] for s in students if s["face_encoding"] is not None]
    student_ids = [s["id"] for s in students if s["face_encoding"] is not None]

    if not face_encodings:
        print("[Faiss] Không tìm thấy face encoding hợp lệ trong CSDL.")
        return

    # Chuyển đổi danh sách thành một mảng NumPy
    encodings_matrix = np.array(face_encodings).astype('float32')
    
    # Chuẩn hóa các vector (quan trọng để so sánh cosine)
    faiss.normalize_L2(encodings_matrix)

    # Lấy số chiều của vector (ví dụ: 512)
    d = encodings_matrix.shape[1]

    # Tạo chỉ mục Faiss. IndexFlatL2 là chỉ mục đơn giản nhất, thực hiện tìm kiếm chính xác.
    index = faiss.IndexFlatL2(d)
    
    # Thêm các vector vào chỉ mục
    index.add(encodings_matrix)

    print(f"[Faiss] Đã xây dựng xong chỉ mục với {index.ntotal} vector.")

    # --- Lưu chỉ mục và file ánh xạ ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, FAISS_INDEX_FILE)
    mapping_path = os.path.join(base_dir, ID_MAPPING_FILE)
    
    faiss.write_index(index, index_path)
    with open(mapping_path, 'w') as f:
        json.dump(student_ids, f)
        
    print(f"[Faiss] Đã lưu chỉ mục vào '{FAISS_INDEX_FILE}' và ánh xạ vào '{ID_MAPPING_FILE}'.")


def load_index():
    """
    Tải chỉ mục Faiss và file ánh xạ ID từ file.
    Nếu file không tồn tại, gọi hàm build_and_save_index().
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, FAISS_INDEX_FILE)
    mapping_path = os.path.join(base_dir, ID_MAPPING_FILE)

    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        print("[Faiss] Không tìm thấy file chỉ mục. Bắt đầu xây dựng lại từ đầu...")
        build_and_save_index()

    try:
        print("[Faiss] Đang tải chỉ mục...")
        index = faiss.read_index(index_path)
        with open(mapping_path, 'r') as f:
            id_mapping = json.load(f)
        print(f"[Faiss] Tải thành công chỉ mục với {index.ntotal} vector.")
        return index, id_mapping
    except Exception as e:
        print(f"[Faiss Lỗi] Không thể tải chỉ mục: {e}. Thử xây dựng lại.")
        build_and_save_index()
        # Thử tải lại lần nữa sau khi xây dựng
        try:
            index = faiss.read_index(index_path)
            with open(mapping_path, 'r') as f:
                id_mapping = json.load(f)
            return index, id_mapping
        except Exception as final_e:
            print(f"[Faiss Lỗi nghiêm trọng] Không thể tải chỉ mục ngay cả sau khi xây dựng lại: {final_e}")
            return None, None
    
def add_to_index(student_id, face_encoding):
    """
    Thêm 1 face_encoding và student_id vào Faiss index và cập nhật file ánh xạ.
    """
    if face_encoding is None or student_id is None:
        print("[Faiss] Thông tin không hợp lệ, không thể thêm vào chỉ mục.")
        return

    # Đọc index hiện tại hoặc tạo mới
    index, id_mapping = load_index()
    if index is None:
        print("[Faiss] Tạo mới chỉ mục Faiss.")
        d = len(face_encoding)
        index = faiss.IndexFlatL2(d)
        id_mapping = []

    # Chuẩn hóa và thêm vector
    face_vector = np.array([face_encoding], dtype='float32')
    faiss.normalize_L2(face_vector)
    index.add(face_vector)
    id_mapping.append(student_id)

    # Lưu lại
    _save_index_and_mapping(index, id_mapping)
    print(f"[Faiss] Đã thêm 1 vector vào chỉ mục. Tổng số: {index.ntotal} vector.")

def _save_index_and_mapping(index, id_mapping):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, FAISS_INDEX_FILE)
    mapping_path = os.path.join(base_dir, ID_MAPPING_FILE)

    faiss.write_index(index, index_path)
    with open(mapping_path, 'w') as f:
        json.dump(id_mapping, f)

def remove_from_index(student_id):
    """
    Xóa face_encoding và student_id khỏi Faiss index và file ánh xạ.
    """
    index, id_mapping = load_index()
    if index is None or id_mapping is None:
        print("[Faiss] Không thể tải chỉ mục để xóa.")
        return

    try:
        remove_idx = id_mapping.index(student_id)
    except ValueError:
        print(f"[Faiss] Không tìm thấy student_id {student_id} trong ánh xạ.")
        return

    # Xóa vector bằng cách tạo lại index mới trừ đi phần tử đó
    vectors = index.reconstruct_n(0, index.ntotal)
    vectors = np.delete(vectors, remove_idx, axis=0)
    id_mapping.pop(remove_idx)

    if len(vectors) == 0:
        print("[Faiss] Đã xóa hết vector, làm trống chỉ mục.")
        d = index.d
        index = faiss.IndexFlatL2(d)
    else:
        faiss.normalize_L2(vectors)
        d = vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(vectors)

    _save_index_and_mapping(index, id_mapping)
    print(f"[Faiss] Đã xóa 1 vector. Còn lại: {index.ntotal} vector.")
