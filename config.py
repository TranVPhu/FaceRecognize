# config.py
"""
Lưu trữ tất cả các hằng số và cài đặt cho ứng dụng.
"""

# Cài đặt tối ưu cho máy yếu
SKIP_FRAMES = 10 # Giảm số lần xử lý khung hình để tăng tốc độ
RESIZE_FACTOR = 0.9 # Giảm kích thước ảnh để xử lý nhanh hơn
RECOGNITION_TOLERANCE = 0.6 # Ngưỡng nhận diện khuôn mặt

# Cài đặt cơ sở dữ liệu
DB_NAME = 'student_faces.db'