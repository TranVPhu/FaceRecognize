import sqlite3
import os
import numpy as np
from config import DB_NAME 

def get_db_path(db_name=DB_NAME) -> str:
    """Trả về đường dẫn tuyệt đối tới file database trong thư mục chứa file Python"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, db_name)

def create_table():
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY NOT NULL UNIQUE,
                name TEXT NOT NULL,
                gender TEXT,
                dob TEXT,                     -- Ngày sinh (dạng chuỗi)
                class TEXT NOT NULL,
                stt INTEGER,
                image_path TEXT,              -- Đường dẫn ảnh đại diện
                face_encoding BLOB NOT NULL UNIQUE,  -- Sử dụng BLOB thay vì TEXT
                school_year TEXT
            )
        ''')
        # Tạo chỉ mục cho face_encoding
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_encoding ON students (face_encoding)')
        conn.commit()
        print(f"[DB] Đã kiểm tra/tạo bảng 'students' và chỉ mục 'idx_face_encoding'.")

def get_all_students():
    """Lấy tất cả học sinh từ CSDL."""
    try:
        with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, dob, class, gender, school_year, stt, image_path, face_encoding FROM students")
            students = []
            while True:
                rows = cursor.fetchmany(100)
                if not rows:
                    break
                for row in rows:
                    face_encoding = None
                    if row["face_encoding"]:
                        if isinstance(row["face_encoding"], bytes):
                            face_encoding = np.frombuffer(row["face_encoding"], dtype=np.float64)
                        else:
                            print(f"[DB Cảnh báo] face_encoding không phải bytes cho ID={row['id']}, loại: {type(row['face_encoding'])}")
                    students.append({
                        "id": row["id"],
                        "name": row["name"],
                        "dob": row["dob"],
                        "class": row["class"],
                        "gender": row["gender"],
                        "school_year": row["school_year"],
                        "stt": row["stt"],
                        "image_path": row["image_path"],
                        "face_encoding": face_encoding
                    })
        return students
    except sqlite3.Error as e:
        print(f"[DB Lỗi] Không thể lấy danh sách học sinh: {e}")
        return []

def add_student(student_id,name, dob, student_class, face_encoding, gender, school_year, stt, image_path=None):
    """Thêm một học sinh mới vào CSDL và trả về ID của học sinh đó."""
    try:
        if not student_id or not student_id.strip():
            raise ValueError("ID học sinh không được để trống.")
        if face_encoding is None or not isinstance(face_encoding, np.ndarray):
            raise ValueError("face_encoding phải là np.ndarray và không được là None")
        with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.cursor()
            encoding_blob = face_encoding.astype(np.float64).tobytes()

            cursor.execute("""
                INSERT INTO students (id, name, dob, class, face_encoding, gender, school_year, stt, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (student_id, name, dob, student_class, encoding_blob, gender, school_year, stt, image_path))
            conn.commit()
            return cursor.lastrowid # Trả về ID
    except sqlite3.Error as e:
        print(f"[DB Lỗi] Không thể thêm học sinh {student_id}: {e}")
        return None # Trả về None nếu lỗi

def update_student(student_id, new_name, new_dob, new_class, new_gender, new_school_year, new_stt, new_image_path, face_encoding=None):
    """Cập nhật tên và lớp cho học sinh dựa trên ID."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()     
            query = (
                "UPDATE students SET name = ?, dob = ?, class = ?, gender = ?, "
                "school_year = ?, stt = ?, image_path = ?"
            )
            params = [
                new_name,
                new_dob,
                new_class,
                new_gender,
                new_school_year,
                new_stt,
                new_image_path,
            ]
            if face_encoding is not None:
                encoding_blob = face_encoding.astype(np.float64).tobytes()
                query += ", face_encoding = ?"
                params.append(encoding_blob)

            query += " WHERE id = ?"
            params.append(student_id)
            
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0 # Trả về True nếu có hàng được cập nhật
    except sqlite3.Error as e:
        print(f"[DB Lỗi] Không thể cập nhật học sinh ID={student_id}: {e}")
        return False

def delete_student(student_id):
    """Xóa học sinh theo ID khỏi CSDL và ảnh nếu có."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Lấy thông tin trước khi xóa để xóa ảnh
            cursor.execute("SELECT image_path FROM students WHERE id = ?", (student_id,))
            row = cursor.fetchone()

            if row and row["image_path"]:
                image_path = row["image_path"]
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                except OSError as e:
                    print(f"[DB Lỗi] Không thể xóa ảnh {image_path}: {e}")

            # Thực hiện xóa
            cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
            conn.commit()

        return True
    except sqlite3.Error as e:
        print(f"[DB Lỗi] Không thể xóa học sinh ID={student_id}: {e}")
        return False

db_path = get_db_path()

# def get_student_by_id(student_id):
#     """Lấy thông tin chi tiết của học sinh theo ID."""
#     try:
#         with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
#             conn.row_factory = sqlite3.Row
#             cursor = conn.cursor()
#             cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
#             row = cursor.fetchone()

#         if row:
#             encoding = None
#             if row["face_encoding"]:
#                 if isinstance(row["face_encoding"], bytes):
#                     encoding = np.frombuffer(row["face_encoding"], dtype=np.float64)
#                 else:
#                     print(f"[DB Cảnh báo] face_encoding không phải bytes cho ID={student_id}, loại: {type(row['face_encoding'])}")
#             return {
#                 "id": row["id"],
#                 "name": row["name"],
#                 "dob": row["dob"] if "dob" in row.keys() and row["dob"] is not None else "",
#                 "gender": row["gender"] if "gender" in row.keys() and row["gender"] is not None else "",
#                 "class": row["class"] if "class" in row.keys() and row["class"] is not None else "",
#                 "image_path": row["image_path"] if "image_path" in row.keys() and row["image_path"] is not None else "",
#                 "encoding": encoding
#             }
#         return None
#     except sqlite3.Error as e:
#         print(f"[DB Lỗi] Không thể lấy học sinh ID={student_id}: {e}")
#         return None
