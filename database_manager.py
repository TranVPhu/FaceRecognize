# database_manager.py
"""
Module 2: Chịu trách nhiệm cho mọi thao tác với cơ sở dữ liệu SQLite.
Bao gồm: Thêm, sửa, xóa, và truy xuất danh sách học sinh.
"""
import sqlite3
import json
import numpy as np
from config import DB_NAME

def create_table():
    """Tạo bảng students nếu chưa tồn tại."""
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
    print(f"[DB] Đã kiểm tra/tạo bảng 'students'.")

def get_all_students():
    """Lấy tất cả học sinh từ CSDL."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row # Giúp truy cập cột bằng tên
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, class, face_encoding FROM students")
            rows = cursor.fetchall()

        students = []
        for row in rows:
            encoding = np.array(json.loads(row["face_encoding"]))
            students.append({
                "id": row["id"],
                "name": row["name"],
                "class": row["class"],
                "encoding": encoding
            })
        return students
    except sqlite3.Error as e:
        print(f"[DB Lỗi] Không thể lấy danh sách học sinh: {e}")
        return []

def add_student(name, student_class, face_encoding):
    """Thêm một học sinh mới vào CSDL."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            encoding_json = json.dumps(face_encoding.tolist())
            cursor.execute("INSERT INTO students (name, class, face_encoding) VALUES (?, ?, ?)",
                           (name, student_class, encoding_json))
        return True
    except sqlite3.IntegrityError:
        print(f"[DB Lỗi] Dữ liệu khuôn mặt hoặc thông tin bị trùng lặp.")
        return False

def update_student(student_id, new_name, new_class):
    """Cập nhật tên và lớp cho học sinh dựa trên ID."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE students SET name = ?, class = ? WHERE id = ?",
                       (new_name, new_class, student_id))
        return cursor.rowcount > 0 # Trả về True nếu có hàng được cập nhật

def delete_student(student_id):
    """Xóa học sinh dựa trên ID."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
        return cursor.rowcount > 0 # Trả về True nếu có hàng bị xóa