# add_student_dialog.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QPushButton, QFileDialog,
                             QLabel, QHBoxLayout, QMessageBox, QComboBox, QFormLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import os
from datetime import datetime
import database_manager as db

class AddStudentDialog(QDialog):
    def __init__(self, frame=None, face_encoding=None, existing_data=None, parent=None):
        super().__init__(parent)
        if existing_data:
            self.setWindowTitle("Sửa thông tin học sinh")
        else:
            self.setWindowTitle("Thêm học sinh mới")
        self.setFixedSize(400, 550)

        self.new_image = None
        self.captured_image = frame
        self.face_encoding = face_encoding
        self.image_path = None
        self.student_data = None

        # ===== Layout chính =====
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # ===== Khung ảnh =====
        self.image_label = QLabel("Vui lòng chụp ảnh hoặc chọn ảnh từ thư mục")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(380, 250)
        layout.addWidget(self.image_label)

        self.browse_button = QPushButton("Thay đổi ảnh")
        self.browse_button.clicked.connect(self.change_image_from_file)
        layout.addWidget(self.browse_button)

        # ===== Form nhập liệu =====
        form_layout = QFormLayout()
        layout.addLayout(form_layout)

        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("Mã CCCD...")
        form_layout.addRow("Mã học sinh:", self.code_input)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Họ và tên...")
        form_layout.addRow("Họ và tên:", self.name_input)

        self.gender_input = QComboBox()
        self.gender_input.addItems(["Nam", "Nữ"])
        form_layout.addRow("Giới tính:", self.gender_input)

        self.dob_input = QLineEdit()
        self.dob_input.setPlaceholderText("dd/mm/yyyy")
        form_layout.addRow("Ngày sinh:", self.dob_input)
        # Gắn xử lý tự động chèn dấu '/'
        self.dob_input.textChanged.connect(self.format_dob_input)

        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Lớp...")
        form_layout.addRow("Lớp:", self.class_input)

        self.index_input = QLineEdit()
        self.index_input.setPlaceholderText("Số thứ tự...")
        form_layout.addRow("Số thứ tự:", self.index_input)

        self.year_input = QComboBox()
        self.year_input.addItems(["2023-2024", "2024-2025", "2025-2026"])
        form_layout.addRow("Năm học:", self.year_input)

        # ===== Buttons =====
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Lưu")
        self.cancel_button = QPushButton("Hủy")
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.save_button.clicked.connect(self.save_data)
        self.cancel_button.clicked.connect(self.reject)

        # ===== Nếu đang sửa: điền thông tin =====
        if existing_data:
            self.populate_fields(existing_data)
            self.code_input.setText(str(existing_data.get('id', '')))
            self.code_input.setReadOnly(True) # Không cho sửa ID khi đang edit
        if self.captured_image is not None:
            self.display_image_from_array(self.captured_image)
            self.browse_button.setVisible(False)
        else:
            self.image_path = existing_data.get("image_path") if existing_data else None
            if self.image_path and os.path.exists(self.image_path):
                self.display_image_from_path(self.image_path)
            self.browse_button.setVisible(True)

    def populate_fields(self, data: dict):
        """Điền dữ liệu sẵn vào form nếu đang chỉnh sửa."""
        self.code_input.setText(str(data.get("code", "")))
        self.name_input.setText(data.get("name", ""))
        self.class_input.setText(data.get("class", ""))
        self.index_input.setText(str(data.get("stt", "")))
        self.image_path = data.get("image_path", None)
        self.dob_input.setText(data.get("dob", ""))  

        # Chọn mục trong combobox nếu trùng
        year = data.get("school_year", "")
        gender = data.get("gender", "")

        year_index = self.year_input.findText(year)
        if year_index >= 0:
            self.year_input.setCurrentIndex(year_index)

        gender_index = self.gender_input.findText(gender)
        if gender_index >= 0:
            self.gender_input.setCurrentIndex(gender_index)

        # Hiển thị ảnh nếu có đường dẫn hợp lệ
        if self.image_path and os.path.exists(self.image_path):
            self.display_image_from_path(self.image_path)

    def format_dob_input(self, text):
        # Xóa ký tự không phải số hoặc /
        digits = ''.join(c for c in text if c.isdigit())

        if len(digits) >= 2:
            day = digits[:2]
        else:
            day = digits

        if len(digits) >= 4:
            month = digits[2:4]
        elif len(digits) > 2:
            month = digits[2:]
        else:
            month = ""

        if len(digits) > 4:
            year = digits[4:8]
        else:
            year = ""

        new_text = day
        if len(digits) > 2:
            new_text += "/" + month
        if len(digits) > 4:
            new_text += "/" + year

        # Ngắt kết nối tạm thời để tránh lặp vô hạn
        self.dob_input.blockSignals(True)
        self.dob_input.setText(new_text)
        self.dob_input.blockSignals(False)

        # Đặt con trỏ cuối
        self.dob_input.setCursorPosition(len(new_text))
            
    def change_image_from_file(self):        
        file_name, _ = QFileDialog.getOpenFileName(self, "Mở Ảnh", "", "Image Files (*.jpg *.png *.jpeg)")
        if not file_name:
            return
        
        capture_img = cv2.imread(file_name)
        if capture_img is None:
            QMessageBox.warning(self, "Lỗi", "Không thể mở ảnh đã chọn.")
            return
        
        scaled_image = cv2.resize(capture_img, (320, 320))
        code = self.code_input.text().strip()
        filename = f"{code}.jpg"
        # Lấy thư mục dự án (hoặc thư mục chứa DB)
        base_dir = os.path.dirname(db.get_db_path())

        # Nếu muốn lưu trong thư mục con "images"
        images_dir = os.path.join(base_dir, "images")

        # Tạo đường dẫn ảnh đầy đủ
        image_path = os.path.join(images_dir, filename)

        if scaled_image is not None:
            os.remove(image_path)
            cv2.imwrite(image_path, scaled_image)
        self.display_image_from_array(scaled_image)
        self.new_image = capture_img

    def display_image_from_array(self, img):
        #current_img = cv2.imread(img)
        if img is None:
            #logging.error(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
            QMessageBox.warning(self, "Lỗi", "Không thể mở ảnh. Vui lòng kiểm tra đường dẫn.")
            return
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image).scaled(380, 250, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def display_image_from_path(self, path):
        pixmap = QPixmap(path).scaledToWidth(300)
        self.image_label.setPixmap(pixmap)
    
    def save_data(self):
        name = self.name_input.text().strip()
        lop = self.class_input.text().strip()
        nam_hoc = self.year_input.currentText()
        try:
            stt = int(self.index_input.text().strip())
            code = self.code_input.text().strip()
        except ValueError:
            QMessageBox.warning(self, "Lỗi", "Mã định danh, Số thứ tự phải là số nguyên.")
            return
        gender_text = self.gender_input.currentText()
        
        dob = self.dob_input.text().strip()
        try:
            valid = datetime.strptime(dob, "%d/%m/%Y")
        except ValueError:
            QMessageBox.warning(self, "Lỗi", "Ngày sinh không hợp lệ. Vui lòng nhập đúng định dạng dd/mm/yyyy.")
            return
        
        if not code or not name or not lop or not stt or not dob:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập đầy đủ các trường:\n- Mã học sinh\n- Họ và tên\n- Lớp\n- Số thứ tự\n- Ngày sinh")
            return
        
        # Đặt tên file
        filename = f"{code}.jpg"
        # Lấy thư mục dự án (hoặc thư mục chứa DB)
        base_dir = os.path.dirname(db.get_db_path())

        # Nếu muốn lưu trong thư mục con "images"
        images_dir = os.path.join(base_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Tạo đường dẫn ảnh đầy đủ
        image_path = os.path.join(images_dir, filename)

        save_dir = image_path

        # Nếu có ảnh, resize và lưu
        if self.captured_image is not None:
            scaled_image = cv2.resize(self.captured_image, (320, 320))
            cv2.imwrite(save_dir, scaled_image)
            self.image_path = save_dir

        # Lưu thông tin
        self.student_data = {
            "code": code,
            "name": name,
            "class": lop,
            "dob": dob,
            "school_year": nam_hoc,
            "stt": stt,
            "gender": gender_text,
            "image": self.captured_image,
            "new_image": self.new_image,
            "image_path": self.image_path,
            "face_encoding": self.face_encoding
        }
        self.accept()

    def get_data(self):
        return self.student_data
