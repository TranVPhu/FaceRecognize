import cv2
import mediapipe as mp
import sqlite3
import numpy as np

# Khởi tạo BlazeFace từ MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Kết nối đến cơ sở dữ liệu SQLite
conn = sqlite3.connect('face_database.db')
cursor = conn.cursor()

# Tạo bảng nếu chưa tồn tại
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        face_encoding BLOB,
        age INTEGER,
        gender TEXT,
        address TEXT
    )
''')
conn.commit()

def encode_face(image):
    """Mã hóa khuôn mặt thành vector đặc trưng"""
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        keypoints = detection.location_data.relative_keypoints
        # Tạo vector đặc trưng từ bounding box và keypoints
        encoding = [
            bbox.xmin, bbox.ymin, bbox.width, bbox.height,
            keypoints[0].x, keypoints[0].y,  # Mắt phải
            keypoints[1].x, keypoints[1].y,  # Mắt trái
            keypoints[2].x, keypoints[2].y,  # Mũi
            keypoints[3].x, keypoints[3].y,  # Miệng
            keypoints[4].x, keypoints[4].y,  # Tai phải
            keypoints[5].x, keypoints[5].y   # Tai trái
        ]
        return np.array(encoding, dtype=np.float32)
    return None

def insert_user(name, image, age, gender, address):
    """Thêm người dùng mới vào cơ sở dữ liệu"""
    encoding = encode_face(image)
    if encoding is not None:
        cursor.execute('''
            INSERT INTO users (name, face_encoding, age, gender, address)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, encoding.tobytes(), age, gender, address))
        conn.commit()
        return True
    return False

def recognize_face(image):
    """Nhận diện khuôn mặt và tra cứu thông tin"""
    encoding = encode_face(image)
    if encoding is None:
        return None

    cursor.execute('SELECT id, name, face_encoding, age, gender, address FROM users')
    users = cursor.fetchall()
    
    min_distance = float('inf')
    matched_user = None
    
    for user in users:
        user_id, name, stored_encoding, age, gender, address = user
        stored_encoding = np.frombuffer(stored_encoding, dtype=np.float32)
        
        # Tính khoảng cách Euclidean
        distance = np.linalg.norm(encoding - stored_encoding)
        if distance < min_distance and distance < 0.1:  # Ngưỡng nhận diện
            min_distance = distance
            matched_user = {
                'id': user_id,
                'name': name,
                'age': age,
                'gender': gender,
                'address': address
            }
    
    return matched_user

def main():
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Nhận diện khuôn mặt
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                
                # Vẽ hộp bao quanh khuôn mặt
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Tra cứu thông tin
                user_info = recognize_face(frame[y:y+height, x:x+width])
                if user_info:
                    info_text = f"Name: {user_info['name']}, Age: {user_info['age']}, Gender: {user_info['gender']}"
                    cv2.putText(frame, info_text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        # Nhấn 'q' để thoát, nhấn 's' để lưu người dùng mới
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            name = input("Enter name: ")
            age = int(input("Enter age: "))
            gender = input("Enter gender: ")
            address = input("Enter address: ")
            if insert_user(name, frame, age, gender, address):
                print("User saved successfully!")
            else:
                print("Failed to save user!")
    
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()