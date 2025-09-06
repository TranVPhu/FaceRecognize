# file: config.py

import psutil

def get_optimal_config():
    # Lấy số nhân vật lý của CPU và tổng RAM (GB)
    cpu_count = psutil.cpu_count(logical=False) # Ưu tiên nhân vật lý
    if cpu_count is None: # Fallback nếu không lấy được nhân vật lý
        cpu_count = psutil.cpu_count(logical=True)
        
    mem_gb = psutil.virtual_memory().total / (1024 ** 3)

    # Cấu hình cho máy cấu hình cao (High-end)
    if mem_gb >= 16 and cpu_count >= 8:
        print("[Config] Chế độ cấu hình: High-end")
        return {
            "SKIP_FRAMES": 5,
            "RESIZE_FACTOR": 0.7,
            "DET_SIZE": (320, 320),
            "MAX_WORKERS": 4  # Dùng nhiều luồng hơn
        }
    # Cấu hình cho máy tầm trung (Mid-range)
    elif mem_gb >= 8 and cpu_count >= 4:
        print("[Config] Chế độ cấu hình: Mid-range")
        return {
            "SKIP_FRAMES": 10,
            "RESIZE_FACTOR": 0.6,
            "DET_SIZE": (320, 320),
            "MAX_WORKERS": 2  # Mặc định
        }
    # Cấu hình cho máy cấu hình thấp (Low-end)
    else:
        print("[Config] Chế độ cấu hình: Low-end")
        return {
            "SKIP_FRAMES": 15,
            "RESIZE_FACTOR": 0.4,
            "DET_SIZE": (160, 160),
            "MAX_WORKERS": 1  # Chỉ dùng 1 luồng để tiết kiệm tài nguyên
        }

# Lấy cấu hình và gán vào các biến
config = get_optimal_config()
SKIP_FRAMES = config["SKIP_FRAMES"]
RESIZE_FACTOR = config["RESIZE_FACTOR"]
DET_SIZE = config["DET_SIZE"]
MAX_WORKERS = config["MAX_WORKERS"] # Thêm biến mới

# Các hằng số khác
RECOGNITION_TOLERANCE = 0.6
MOTION_THRESHOLD = 25
DB_NAME = 'student_faces.db'

# Danh sách thuật toán phát hiện khuôn mặt
FACE_DETECTION_ALGORITHMS = [
    "Haar Cascade",
    "MTCNN",
    "Dlib HOG"
]

# Danh sách thuật toán nhận diện khuôn mặt
FACE_RECOGNITION_ALGORITHMS = [
    "ArcFace",
    "FaceNet",
    "OpenFace"
]