# import site
# print(site.getsitepackages())

# import face_recognition
# print(dir(face_recognition))

# import face_recognition
# print("face_recognition imported successfully!")

import dlib
import os

# Lấy đường dẫn đến thư mục cài đặt của dlib
dlib_path = os.path.dirname(dlib.__file__)
print(f"Thư mục cài đặt của dlib: {dlib_path}")

# Các tệp mô hình thường nằm trong thư mục con 'models' hoặc tương tự
# Bạn có thể thử kiểm tra các đường dẫn phổ biến
model_path_1 = os.path.join(dlib_path, 'models')
model_path_2 = os.path.join(dlib_path, '..', 'face_recognition_models') # Đôi khi nằm ở cấp cao hơn

print(f"Kiểm tra đường dẫn mô hình tiềm năng 1: {model_path_1}")
print(f"Kiểm tra đường dẫn mô hình tiềm năng 2: {model_path_2}")

# Bạn có thể kiểm tra xem các tệp mô hình có tồn tại ở đó không
# Ví dụ: shape_predictor_68_face_landmarks.dat, dlib_face_recognition_resnet_model_v1.dat
