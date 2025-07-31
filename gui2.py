import sys
import gc
import os
import pynvml
import psutil
import logging
import uuid
import faiss_manager
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, 
                             QPushButton, QListWidget, QGroupBox, QHBoxLayout, 
                             QFileDialog, QMessageBox, QListWidgetItem, QGridLayout,
                             QStatusBar, QStyle, QDialog, QSlider,
                             QStyleOptionSlider)
from PyQt5.QtCore import Qt, QSize, QTimer, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
import database_manager as db
from face_processor import FaceProcessor
from add_student_dialog import AddStudentDialog
from config import RESIZE_FACTOR
from scaler import FixedScaler
from Video_Thread import VideoThread
from time import time

class CustomSlider(QSlider):
    """
    Lớp QSlider tùy chỉnh cho phép click chuột để nhảy đến vị trí mong muốn.
    """
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Tính toán giá trị mới dựa trên vị trí click chuột
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            
            # Lấy vị trí của rãnh trượt
            groove_rect = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderGroove, self)
            # Lấy vị trí của con trượt
            handle_rect = self.style().subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)

            # Đảm bảo click nằm ngoài con trượt hiện tại
            if not handle_rect.contains(event.pos()):
                # Chuyển đổi vị trí click sang giá trị của slider
                if self.orientation() == Qt.Horizontal:
                    # Công thức tính giá trị dựa trên vị trí x của chuột
                    new_val = QStyle.sliderValueFromPosition(
                        self.minimum(), 
                        self.maximum(), 
                        event.x() - groove_rect.x(), 
                        groove_rect.width(), 
                        option.upsideDown
                    )
                else:
                    # Tương tự cho slider dọc
                    new_val = QStyle.sliderValueFromPosition(
                        self.minimum(), 
                        self.maximum(), 
                        groove_rect.height() - (event.y() - groove_rect.y()), 
                        groove_rect.height(), 
                        option.upsideDown
                    )
                
                # Đặt giá trị mới và phát tín hiệu
                self.setValue(new_val)
                self.sliderReleased.emit() # Phát tín hiệu để tua video ngay lập tức
        
        # Gọi lại sự kiện gốc để giữ các chức năng khác
        super().mousePressEvent(event)

# Cửa sổ chính của ứng dụng
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ thống Nhận diện Khuôn mặt")
        self.setGeometry(100, 100, 1280, 720)

        # --- Bước 1: Khởi tạo các đối tượng xử lý và dữ liệu ---
        self.scaler = FixedScaler(target_width=640)

        # Tải chỉ mục Faiss và ánh xạ ID
        self.faiss_index, self.id_mapping = faiss_manager.load_index()

        self.known_students = db.get_all_students()
        self.known_students_dict = {s['id']: s for s in self.known_students}
        
        # Khởi tạo FaceProcessor với chỉ mục Faiss
        self.face_processor = FaceProcessor(self.faiss_index, self.id_mapping)

        # --- Bước 2: Khởi tạo các biến trạng thái của ứng dụng ---
        self.thread = None
        self.current_frame = None  # Đổi tên từ current_frame_cv cho nhất quán
        self.recognition_results = []
        self.selected_student_id = None
        self.last_results = []

        # --- Bước 3: Khai báo trước các biến sẽ chứa Widget (được tạo trong init_ui) ---
        self.image_label = None
        self.student_list_widget = None
        
        # Widget điều khiển video và các thành phần
        self.video_controls_widget = None
        self.play_pause_button = None
        self.video_slider = None
        self.time_label = None
        
        # Nút sửa thông tin
        self.add_student_button = None
        self.edit_student_button = None
        self.delete_button = None

        # --- Bước 3: Biến để lưu trữ FPS và thông tin tài nguyên ---
        self.current_fps = 0.0
        self.resource_info = {
            "cpu_usage": 0.0,
            "ram_used": 0.0,
            "ram_total": 0.0,
            "ram_usage": 0.0,
            "gpu_usage": 0.0,
            "gpu_memory_used": 0.0,
            "gpu_memory_total": 0.0
        }
        # --- Bước 5: Khởi tạo thanh trạng thái ---
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sẵn sàng")

        # Khởi tạo timer để theo dõi tài nguyên
        self.resource_timer = QTimer(self)
        self.resource_timer.timeout.connect(self.update_resource_usage)
        self.resource_timer.start(2000)  # Cập nhật mỗi 2 giây
        
        # Khởi tạo pynvml để theo dõi GPU
        self.gpu_available = False
        # try:
        #     pynvml.nvmlInit()
        #     self.gpu_available = True
        # except pynvml.NVMLError:
        #     logging.warning("Không thể khởi tạo pynvml. Theo dõi GPU bị tắt.")

        # --- Bước 4: Gọi các hàm dựng giao diện ---
        self.init_ui()

    def init_ui(self):
        """Khởi tạo giao diện người dùng với bố cục giữ nguyên, và thanh điều khiển bằng chiều rộng video."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_v_layout = QVBoxLayout(central_widget)
        top_h_layout = QHBoxLayout()

        # -------------------------
        # 📷 Cột trái: Ảnh/video + điều khiển video (gộp vào layout dọc)
        # -------------------------
        left_v_layout = QVBoxLayout()

        self.image_label = QLabel("Mở nguồn video để bắt đầu")
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)

        # 👉 Cho phép điều chỉnh kích thước ảnh (có thể cố định nếu muốn)
        self.image_label.setFixedWidth(640)
        self.image_label.setFixedHeight(480)

        left_v_layout.addWidget(self.image_label)

        # 🎚️ Thanh điều khiển video nằm ngay dưới video
        video_controls = self.create_video_controls()
        video_controls.setVisible(False)  # Ẩn mặc định
        video_controls.setFixedWidth(self.image_label.width())  # ✅ đặt chiều rộng bằng ảnh
        self.video_controls_widget = video_controls

        left_v_layout.addWidget(video_controls)

        top_h_layout.addLayout(left_v_layout, 7)

        # -------------------------
        # 📋 Cột phải: điều khiển khác
        # -------------------------
        right_panel_v_layout = QVBoxLayout()
        controls_group = self.create_controls_group()
        list_group = self.create_list_group()
        info_group = self.create_info_group()

        right_panel_v_layout.addWidget(controls_group)
        right_panel_v_layout.addWidget(list_group)
        right_panel_v_layout.addWidget(info_group)

        top_h_layout.addLayout(right_panel_v_layout, 3)

        # -------------------------
        # 📐 Tổng layout
        # -------------------------
        main_v_layout.addLayout(top_h_layout)

        # -------------------------
        # 📌 Kết nối sự kiện
        # -------------------------
        self.student_list_widget.itemClicked.connect(self.display_student_info)
        self.image_label.mousePressEvent = self.on_image_click

    def create_list_group(self):
        """Tạo group box cho danh sách học sinh nhận diện."""
        group = QGroupBox("Danh sách nhận diện")
        layout = QVBoxLayout()
        self.student_list_widget = QListWidget()
        
        # --- THAY ĐỔI 1: Áp dụng Style Sheet và đổi con trỏ chuột ---
        self.student_list_widget.setCursor(Qt.PointingHandCursor)
        self.student_list_widget.setStyleSheet("""
            QListWidget::item:selected, QListWidget::item:selected:!active {
                background-color: #3399ff;
                color: white;
                font-weight: bold;
            }
            QListWidget::item:hover {
                background-color: #e0e0e0;
            }
        """)
        # -----------------------------------------------------------
        layout.addWidget(self.student_list_widget)
        group.setLayout(layout)
        self.student_list_widget.setSelectionMode(QListWidget.SingleSelection)  # hoặc ExtendedSelection nếu cần chọn nhiều
        self.student_list_widget.setFocusPolicy(Qt.StrongFocus)
        return group

    def create_info_group(self):
        """Tạo group box cho thông tin chi tiết học sinh."""
        group = QGroupBox("Thao tác")
        layout = QVBoxLayout()
        
        # ✅ THÊM NÚT
        self.add_student_button = QPushButton("Thêm học sinh mới")
        self.add_student_button.clicked.connect(self.add_new_student)
        self.add_student_button.setVisible(False) # Mặc định ẩn
        layout.addWidget(self.add_student_button)

        self.edit_student_button = QPushButton("Sửa thông tin")
        self.edit_student_button.clicked.connect(self.edit_student)
        self.edit_student_button.setVisible(False) # Mặc định ẩn
        layout.addWidget(self.edit_student_button)

        self.delete_button = QPushButton("Xóa học sinh")
        self.delete_button.clicked.connect(self.delete_selected_student)
        self.delete_button.setVisible(False) # Mặc định ẩn
        layout.addWidget(self.delete_button)

        group.setLayout(layout)
        return group

    def create_controls_group(self):
        """Tạo group box cho các nút điều khiển."""
        group = QGroupBox("Nguồn Dữ liệu")
        layout = QGridLayout()
        
        icon_cam = self.style().standardIcon(QStyle.SP_ComputerIcon)
        icon_vid = self.style().standardIcon(QStyle.SP_DriveHDIcon)
        icon_img = self.style().standardIcon(QStyle.SP_DriveNetIcon)

        self.open_cam_button = QPushButton(icon_cam, " Mở Camera")
        self.open_video_button = QPushButton(icon_vid, " Mở Video MP4")
        self.open_image_button = QPushButton(icon_img, " Mở Ảnh")

        layout.addWidget(self.open_cam_button, 0, 0)
        layout.addWidget(self.open_video_button, 0, 1)
        layout.addWidget(self.open_image_button, 0, 2)
        
        group.setLayout(layout)

        self.open_cam_button.clicked.connect(self.start_camera)
        self.open_video_button.clicked.connect(self.browse_video_file)
        self.open_image_button.clicked.connect(self.browse_img_file)
        
        return group

    def create_video_controls(self):
        """Tạo widget chứa các nút điều khiển video."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)

        # Nút Play/Pause
        self.play_pause_button = QPushButton()
        self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_pause_button.clicked.connect(self.toggle_play)

        # Sử dụng CustomSlider
        self.video_slider = CustomSlider(Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setValue(0)
        self.video_slider.sliderReleased.connect(self.seek_video)

        # Nhãn thời gian
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(120)  # Đặt chiều rộng cố định để giao diện gọn gàng
        self.time_label.setAlignment(Qt.AlignCenter)  # Căn giữa để đẹp hơn

        layout.addWidget(self.play_pause_button)
        layout.addWidget(self.video_slider)
        layout.addWidget(self.time_label)

        self.video_controls_widget = widget
        return widget

    def update_video_slider(self, current, total):
        """Cập nhật thanh slider và thời gian video."""
        if not self.thread or not self.thread.cap or not self.thread.cap.isOpened():
            logging.warning("Không thể cập nhật slider: VideoCapture không khả dụng.")
            self.time_label.setText("00:00 / 00:00")
            self.video_slider.setValue(0)
            return

        # Cập nhật giá trị slider
        self.video_slider.setMaximum(total)
        self.video_slider.setValue(current)

        # Lấy FPS từ video, mặc định 30 nếu không hợp lệ
        fps = self.thread.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or not np.isfinite(fps):
            fps = 30
            logging.warning("FPS không hợp lệ, sử dụng mặc định 30.")

        # Tính thời gian
        current_time = current / fps
        total_time = total / fps
        current_min = int(current_time // 60)
        current_sec = int(current_time % 60)
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)

        # Cập nhật nhãn thời gian
        self.time_label.setText(f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}")
        logging.debug(f"Cập nhật slider: frame {current}/{total}, thời gian {current_min:02d}:{current_sec:02d}/{total_min:02d}:{total_sec:02d}")

    def toggle_play(self):
        # Nếu thread không còn chạy, phát lại từ đầu nếu là video
        if not self.thread or not self.thread.isRunning():
            if hasattr(self, 'last_video_file') and isinstance(self.last_video_file, str):
                self.play_video(self.last_video_file)
            return

        # Nếu thread đang chạy, toggle pause
        self.thread.toggle_pause()
        icon = self.style().standardIcon(
            QStyle.SP_MediaPause if not self.thread._is_paused else QStyle.SP_MediaPlay
        )
        self.play_pause_button.setIcon(icon)

    def play_video(self, file_path):
        """Mở video và khởi tạo thời gian ban đầu."""
        self.stop_thread()
        self.clear_student_info()
        self.video_controls_widget.setVisible(True)
        try:
            # Kiểm tra file video trước khi khởi tạo thread
            test_cap = cv2.VideoCapture(file_path)
            if not test_cap.isOpened():
                raise RuntimeError("Không thể mở video.")
            total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or not np.isfinite(fps):
                fps = 30
            total_time = total_frames / fps
            total_min = int(total_time // 60)
            total_sec = int(total_time % 60)
            test_cap.release()

            self.time_label.setText(f"00:00 / {total_min:02d}:{total_sec:02d}")
            self.video_slider.setMaximum(total_frames)
            self.video_slider.setValue(0)

            self.thread = VideoThread(
                        input_source=file_path,
                        known_students=self.known_students,
                        face_processor=self.face_processor, 
                        parent=self
                    )
            self.connect_thread_signals()
            self.thread.start()

        except Exception as e:
            logging.error(f"Lỗi khi mở video: {str(e)}")
            self.show_error_message(f"Lỗi khi mở video: {str(e)}")
            self.time_label.setText("00:00 / 00:00")
            self.video_slider.setValue(0)


    def pause_for_seek(self):
        if self.thread:
            self.thread._is_paused = True

    def seek_to_position(self):
        if self.thread and self.thread.isRunning():
            frame = self.video_slider.value()
            self.thread.seek_to_frame(frame)
            #self.thread._is_paused = False

    def seek_video(self):
        """Lấy giá trị từ slider và yêu cầu thread tua đến frame đó."""
        if self.thread and self.thread.isRunning():
            frame_number = self.video_slider.value()
            self.thread.seek_to_frame(frame_number)
            logging.debug(f"Yêu cầu tua video đến frame {frame_number}")
    
    def display_student_info(self, current, previous=None):
        student_id = current.data(Qt.UserRole)
        self.selected_student_id = student_id
        # Nếu không có ID (ví dụ: click vào "Người lạ"), thì xóa thông tin và ẩn nút Sửa
        if student_id is None:
            self.add_student_button.setVisible(True)
            return
        
        # ✅ Tra cứu trong dictionary
        student = self.known_students_dict.get(student_id)
        
        if student:
            self.edit_student_button.setVisible(True) # ✅ HIỆN NÚT SỬA
            self.delete_button.setVisible(True) # Mặc định ẩn
        else:
            self.clear_student_info() # Gọi hàm xóa để ẩn nút

    # === THÊM HỌC SINH MỚI ===
    def add_new_student(self):
        """Hàm chính điều phối việc thêm học sinh mới."""
        if self.current_frame is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có hình ảnh. Vui lòng mở camera hoặc chọn ảnh trước.")
            return

        # Bước 1: Lấy thông tin từ dialog
        student_data = self._get_data_from_dialog()
        if not student_data:
            return

        # Bước 2: Lấy mã hóa khuôn mặt, Hàm get_single_face_encoding trả về (results, encoding)
        _, face_encoding = self.face_processor.get_single_face_encoding(student_data["image"], self.known_students)
        if face_encoding is None:
            QMessageBox.warning(self, "Lỗi", "Không thể lấy mã khuôn mặt. Vui lòng kiểm tra ảnh đầu vào và đảm bảo chỉ có một khuôn mặt.")
            return

        # Bước 3: Lưu vào CSDL và lấy ID của học sinh mới
        new_student_id = self._save_student_to_db(student_data, face_encoding)
        if not new_student_id:
            return # Thông báo lỗi đã được hiển thị trong hàm con

        # Bước 4: Thêm vào Faiss
        import faiss_manager
        faiss_manager.add_to_index(new_student_id, face_encoding)

        # Bước 5: Cập nhật giao diện
        self._update_ui_after_add(new_student_id)
        QMessageBox.information(self, "Thành công", f"Đã thêm học sinh {student_data['name']}.")

    # === SỬA HỌC SINH ===
    def edit_student(self, student: dict = None, frame=None):
        if self.current_frame is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có hình ảnh. Vui lòng mở camera trước.")
            return

        if self.selected_student_id: 
            student = next((s for s in self.known_students if s['id'] == self.selected_student_id), None)
            if not student:
                return

            dialog = AddStudentDialog(existing_data=student, parent=self)
            result = dialog.exec()
            new_image = None
            if result == QDialog.Accepted:
                updated_data = dialog.get_data()
                new_image = updated_data.pop("new_image", None)
                new_encoding = None
                if new_image is not None:
                    _, new_encoding = self.face_processor.get_single_face_encoding(new_image, [])
                    if new_encoding is None:
                        QMessageBox.warning(self, "Lỗi ảnh", "Không tìm thấy khuôn mặt trong ảnh mới hoặc có quá nhiều khuôn mặt.")
                        return
                    self.current_frame = new_image

                success = db.update_student(
                    self.selected_student_id,
                    updated_data['name'],
                    updated_data['dob'],
                    updated_data['class'],
                    updated_data['gender'],
                    updated_data['school_year'],
                    updated_data['stt'],
                    updated_data['image_path'],
                    face_encoding= new_encoding if new_encoding is not None else student['face_encoding']
                )

                if success:
                    QMessageBox.information(self, "Thành công", "Cập nhật thông tin thành công.")
                    # Lấy lại danh sách mới từ DB
                    list_of_students = db.get_all_students()
                    self.known_students = list_of_students
                    # ✅ Cập nhật lại dictionary để tra cứu nhanh
                    self.known_students_dict = {s['id']: s for s in list_of_students}
                    self.update_results(self.recognition_results)

                    if new_encoding is not None:
                        faiss_manager.remove_from_index(self.selected_student_id)
                        faiss_manager.add_to_index(self.selected_student_id, new_encoding)
                        print(f"✅ Đã cập nhật Faiss cho học sinh ID {self.selected_student_id} (đổi ảnh).")
                    else:
                        print(f"✅ Không đổi ảnh, không cập nhật Faiss.")

                    current_item = self.student_list_widget.currentItem()
                    if current_item:
                        self.display_student_info(current_item)
                else:
                    QMessageBox.warning(self, "Lỗi", "Không thể cập nhật thông tin.")
        
    # --- CÁC HÀM CON TRỢ GIÚP ---
    def _get_data_from_dialog(self):
        """Mở dialog, lấy và xác thực dữ liệu người dùng nhập."""
        dialog = AddStudentDialog(self.current_frame, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return None # Người dùng nhấn Cancel

        student_data = dialog.get_data()
        if not student_data:
            return None

        # Xác thực các trường bắt buộc
        required_fields = ["code","name", "dob", "class", "stt"]
        if not all(student_data.get(field) for field in required_fields):
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập đầy đủ các trường cần thiết.")
            return None

        return student_data

    def _save_student_to_db(self, student_data, face_encoding):
        """Lưu thông tin học sinh vào CSDL và trả về ID."""
        new_id = db.add_student(
            student_id = student_data["code"],
            name=student_data["name"],
            dob=student_data["dob"],
            student_class=student_data["class"],
            face_encoding=face_encoding,
            gender=student_data["gender"],
            school_year=student_data["school_year"],
            stt=student_data["stt"],
            image_path=student_data["image_path"]
        )
        if not new_id:
            QMessageBox.critical(self, "Lỗi", "Không thể thêm học sinh. Mã khuôn mặt có thể đã tồn tại trong CSDL.")
            return None
        return new_id

    def _update_ui_after_add(self, new_student_id):
        """Tải lại dữ liệu và làm mới giao diện sau khi thêm thành công."""
        # 1. Tải lại danh sách học sinh từ CSDL
        self.known_students = db.get_all_students()
        self.known_students_dict = {s['id']: s for s in self.known_students}

        # 2. Chạy lại nhận diện trên frame hiện tại để cập nhật tên
        if self.current_frame is not None:
            # Sử dụng hàm get_single_face_encoding đã tạo ở gợi ý trước
            results, _  = self.face_processor.get_single_face_encoding(self.current_frame, self.known_students)
            self.update_results(results)
            self.update_image(self.current_frame)

        # 3. Tự động chọn học sinh vừa thêm trong danh sách
        for i in range(self.student_list_widget.count()):
            item = self.student_list_widget.item(i)
            if item.data(Qt.UserRole) == new_student_id:
                self.student_list_widget.setCurrentItem(item)
                self.display_student_info(item)
                break

        # (3) Chỉ build lại Faiss nếu thêm học sinh thành công
        print("Cập nhật chỉ mục Faiss sau khi thêm thành công...")
        faiss_manager.build_and_save_index()
        self._reload_faiss_to_faceprocessor()
        print("Đã cập nhật chỉ mục Faiss.")

    def delete_selected_student(self):
        if not self.selected_student_id:
            QMessageBox.information(self, "Thông báo", "Chưa chọn học sinh nào để xóa.")
            return

        confirm = QMessageBox.question(
            self,
            "Xác nhận xóa",
            "Bạn có chắc chắn muốn xóa học sinh này?",
            QMessageBox.Yes | QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            success = db.delete_student(self.selected_student_id)
            if success:
                QMessageBox.information(self, "Thành công", "Đã xóa học sinh.")
                faiss_manager.remove_from_index(self.selected_student_id)
                self.clear_student_info()
                # Lấy lại danh sách mới từ DB
                list_of_students = db.get_all_students()
                self.known_students = list_of_students
                # ✅ Cập nhật lại dictionary để tra cứu nhanh
                self.known_students_dict = {s['id']: s for s in list_of_students}
                self.update_results([])  # Ẩn khuôn mặt cũ
                print("Cập nhật chỉ mục Faiss sau khi xóa thành công...")
                faiss_manager.build_and_save_index()
                self._reload_faiss_to_faceprocessor()
                print("Đã cập nhật chỉ mục Faiss.")
            else:
                QMessageBox.critical(self, "Lỗi", "Không thể xóa học sinh.")
    def _reload_faiss_to_faceprocessor(self):
        self.faiss_index, self.id_mapping = faiss_manager.load_index()
        self.face_processor.faiss_index = self.faiss_index
        self.face_processor.id_mapping = self.id_mapping

    def on_image_click(self, event):
        """
        Xử lý sự kiện click chuột lên ảnh, tìm và chọn học sinh tương ứng.
        Sửa lỗi: Thêm bước map tọa độ từ QLabel sang QPixmap để hoạt động đúng với ảnh tĩnh.
        """
        start_time = time()
        # --- Các bước kiểm tra điều kiện ---
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull() or not self.recognition_results or not hasattr(self, 'scaler'):
            return
        if not self.scaler.scale_ratio > 0:
            return

        # --- Bước 1: Ánh xạ tọa độ click từ QLabel sang QPixmap ---
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()

        # Tọa độ click thực sự trên ảnh (đã tính đến việc ảnh bị co giãn)
        click_x = int(event.x() * pixmap_w / label_w)
        click_y = int(event.y() * pixmap_h / label_h)

        # --- Bước 2: Tính toán scale và lật (giữ nguyên như cũ) ---
        final_scale_factor = self.scaler.scale_ratio / RESIZE_FACTOR
        display_w = pixmap_w

        # --- Bước 3: Duyệt và so sánh ---
        for result in self.recognition_results:
            loc = result.get("location")
            if not loc:
                continue

            # Scale tọa độ box về không gian hiển thị (pixmap)
            top, right, bottom, left = loc
            scaled_top = int(top * final_scale_factor)
            scaled_bottom = int(bottom * final_scale_factor)
            scaled_left = int(left * final_scale_factor)
            scaled_right = int(right * final_scale_factor)

            # Lật tọa độ box để so sánh vì ảnh hiển thị đã bị lật
            flipped_box_left = display_w - scaled_right
            flipped_box_right = display_w - scaled_left

            # So sánh tọa độ click ĐÃ ÁNH XẠ với tọa độ box ĐÃ LẬT
            if flipped_box_left < click_x < flipped_box_right and scaled_top < click_y < scaled_bottom:
                # Tìm item trong danh sách và chọn nó
                for i in range(self.student_list_widget.count()):
                    item = self.student_list_widget.item(i)
                    if item.data(Qt.UserRole) == result["id"]:
                        self.student_list_widget.setCurrentItem(item)
                        self.display_student_info(item)
                        break
                break
        end_time = time()
        response_time = (end_time - start_time) * 1000  # Chuyển sang ms
        logging.info(f"Thời gian phản hồi khi click vào ảnh: {response_time:.2f} ms")
    
    def start_camera(self):
        self.stop_thread()
        self.clear_student_info()
        self.video_controls_widget.setVisible(False) 
        self.thread = VideoThread(
                        input_source=0,
                        known_students=self.known_students,
                        face_processor=self.face_processor, 
                        parent=self
                    )
        self.connect_thread_signals()
        self.thread.start()

    def browse_video_file(self):
        self.stop_thread()
        self.clear_student_info()
        file_name, _ = QFileDialog.getOpenFileName(self, "Mở Video", "", "Video Files (*.mp4 *.avi)")
        print(f"Đường dẫn video: {file_name}")  # Thêm dòng này để kiểm tra
        self.recognition_results = []
        if file_name:
            self.last_video_file = file_name
            self.play_video(self.last_video_file)


    def browse_img_file(self):
        self.stop_thread() 
        file_name, _ = QFileDialog.getOpenFileName(self, "Mở Ảnh", "", "Image Files (*.jpg *.png *.jpeg)")
        if not file_name:
            return

        img = cv2.imread(file_name)
        if img is not None:
            # 1. Dừng luồng video hiện tại (nếu có)
            self.stop_thread()
            self.clear_student_info()
            self.video_controls_widget.setVisible(False)

            # 2. Gọi hàm nhận diện khuôn mặt   
            results, encodings = self.face_processor.get_single_face_encoding(img, self.known_students)

            # 3. Cập nhật trạng thái và gọi trực tiếp các hàm cập nhật GUI
            self.recognition_results = []
            self.current_frame = img
            self.face_encoding = encodings

            self.update_results(results)
            self.update_image(img)
        else:
            QMessageBox.warning(self, "Lỗi", "Không thể đọc tệp ảnh này!")

    def connect_thread_signals(self):
        if not self.thread:
            return
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_results_signal.connect(self.update_results)
        self.thread.finished_signal.connect(self.on_thread_finished)
        self.thread.error_signal.connect(self.show_error_message)
        self.thread.fps_signal.connect(self.update_fps_display)  # Kết nối tín hiệu FPS
        if isinstance(self.thread.input_source, str) and self.thread.total_frames > 1:
            self.thread.progress_signal.connect(self.update_video_slider)
        logging.debug("Đã kết nối các tín hiệu của thread.")

    def show_error_message(self, message):
        QMessageBox.critical(self, "Lỗi", message)

    def update_fps_display(self, fps):
        """Hiển thị FPS trên thanh trạng thái."""
        self.statusBar().showMessage(f"FPS: {fps:.2f}")
    
    def on_thread_finished(self):
        if self.play_pause_button:
            self.play_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        if self.thread and isinstance(self.thread.input_source, str):
            # Nếu là video, reset về đầu và pause
            self.thread._is_paused = True
            self.thread.seek_to_frame(0)  # Tua về frame đầu tiên
            # Đảm bảo thanh điều khiển luôn hiển thị
            if self.video_controls_widget:
                self.video_controls_widget.setVisible(True)
        else:
            # Nếu là camera hoặc ảnh, tắt controls
            if self.video_controls_widget:
                self.video_controls_widget.setVisible(False)


    def update_image(self, img):
        """
        Cập nhật và hiển thị ảnh, sử dụng QPainter cho TOÀN BỘ thao tác vẽ (hộp và chữ).
        Tối ưu hơn bằng cách loại bỏ cv2.rectangle và chỉ lặp qua các kết quả một lần.
        """
        if not isinstance(img, np.ndarray):
            return

        # --- Bước 1: Chuẩn bị ảnh nền ---
        self.current_frame = img  # Lưu ảnh gốc
        display_img = self.scaler.resize_image(img) # Resize ảnh về kích thước hiển thị
        
        # Lật ảnh ngay từ đầu để mọi tọa độ vẽ sau này đều nằm trong không gian đã lật
        flipped_img = cv2.flip(display_img, 1)
        
        # Chuyển ảnh numpy (BGR) sang QImage để QPainter có thể vẽ lên
        h, w, ch = flipped_img.shape
        qt_image = QImage(flipped_img.data, w, h, ch * w, QImage.Format_BGR888)

        # --- Bước 2: Chuẩn bị và thực hiện vẽ ---
        painter = QPainter(qt_image)
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)

        # Tính toán các giá trị cần thiết chỉ một lần
        final_scale_factor = self.scaler.scale_ratio / RESIZE_FACTOR
        display_w = qt_image.width()
        results_to_draw = self.recognition_results if self.recognition_results else self.last_results

        # --- Bước 3: Vòng lặp vẽ duy nhất ---
        for result in results_to_draw:
            loc = result.get("location")
            if not loc:
                continue

            # Scale tọa độ gốc từ không gian nhận diện sang không gian hiển thị
            top, right, bottom, left = loc
            scaled_top = int(top * final_scale_factor)
            scaled_bottom = int(bottom * final_scale_factor)
            scaled_left = int(left * final_scale_factor)
            scaled_right = int(right * final_scale_factor)

            # Xác định màu sắc
            is_stranger = result.get("id") is None
            color = QColor(255, 0, 0) if is_stranger else QColor(0, 255, 0)
            
            pen = QPen(color, 2) # Tạo bút vẽ với màu và độ dày
            painter.setPen(pen)

            # Tính toán tọa độ và kích thước của hộp trong không gian ĐÃ LẬT
            # Tọa độ y không đổi, tọa độ x được lật qua trục giữa của ảnh
            flipped_left = display_w - scaled_right
            rect_w = scaled_right - scaled_left
            rect_h = scaled_bottom - scaled_top
            
            # 1. Vẽ hộp chữ nhật
            painter.drawRect(flipped_left, scaled_top, rect_w, rect_h)

            # 2. Vẽ tên
            # Đo kích thước chữ để căn giữa phía trên hộp
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(result["name"])
            text_x = flipped_left + (rect_w - text_width) // 2
            text_y = scaled_top - 5 # Dịch lên 5 pixel so với cạnh trên của hộp
            
            # Vẽ nền cho chữ để dễ đọc hơn
            painter.fillRect(QRect(text_x - 2, text_y - metrics.ascent(), text_width + 4, metrics.height()), color)
            
            # Đặt lại bút vẽ cho màu chữ (ví dụ: màu trắng)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(text_x, text_y, result["name"])
            
        painter.end() # Kết thúc việc vẽ

        # --- Bước 4: Hiển thị ảnh đã vẽ lên QLabel ---
        new_pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(new_pixmap)
        self.current_pixmap = new_pixmap

    def update_results(self, results):
        """
        Cập nhật danh sách nhận diện. Chỉ vẽ lại widget nếu danh sách ID thay đổi.
        """
        # 1. Trích xuất danh sách ID từ kết quả mới và cũ để so sánh
        # Dùng tuple đã sắp xếp để việc so sánh đáng tin cậy
        new_ids = tuple(sorted(r.get("id") for r in results if r.get("id") is not None))
        old_ids = tuple(sorted(r.get("id") for r in self.recognition_results if r.get("id") is not None))

        # Đếm số người lạ ở mỗi kết quả
        new_strangers = sum(1 for r in results if r.get("id") is None)
        old_strangers = sum(1 for r in self.recognition_results if r.get("id") is None)

        # 2. Nếu danh sách ID và số người lạ không có gì thay đổi, thoát sớm
        if new_ids == old_ids and new_strangers == old_strangers:
            # Cập nhật lại recognition_results để chứa tọa độ mới nhất, nhưng không vẽ lại widget
            self.recognition_results = results
            return

        # --- Chỉ thực hiện các bước dưới đây nếu có sự thay đổi về người trong khung hình ---

        # 3. Cập nhật kết quả và lấy item đang được chọn (nếu có)
        self.recognition_results = results
        current_selected_id = None
        if self.student_list_widget.currentItem():
            current_selected_id = self.student_list_widget.currentItem().data(Qt.UserRole)

        self.student_list_widget.clear()

        for result in results:
            # Phần code còn lại của bạn để tạo và thêm item giữ nguyên
            student_id = result.get("id")
            student_class = ""
            student_name = "Người lạ"
            
            # Lấy thông tin chi tiết nếu là học sinh đã biết
            if student_id is not None and student_id in self.known_students_dict:
                student_info = self.known_students_dict[student_id]
                student_class = student_info.get("class", "")
                student_name = student_info.get("name", "")

            # Định dạng nội dung item
            item_text = (
                "--------------------\n"
                f"ID: {'N/A' if student_id is None else student_id}\n"
                f"Họ và tên: {student_name}\n"
                f"Lớp: {student_class}\n"
                "--------------------"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, student_id)
            item.setSizeHint(QSize(item.sizeHint().width(), 80)) # Điều chỉnh chiều cao

            if student_id is None:
                item.setForeground(Qt.red)

            self.student_list_widget.addItem(item)
            
            # Chọn lại item nếu nó vẫn còn trong danh sách mới
            if current_selected_id is not None and student_id == current_selected_id:
                self.student_list_widget.setCurrentItem(item)

    def clear_student_info(self):
        """Xóa thông tin và ẩn nút Sửa."""
        self.selected_student_id = None # Xóa ID đã lưu
        # ✅ THÊM DÒNG NÀY ĐỂ XÓA HIGHLIGHT
        if self.student_list_widget:
            self.student_list_widget.setCurrentItem(None)

        if self.edit_student_button:
            self.edit_student_button.setVisible(False) # ✅ ẨN NÚT SỬA
            self.add_student_button.setVisible(False) 
            self.delete_button.setVisible(False) # ẨN NÚT XÓA   

    def stop_thread(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait() # Đợi thread kết thúc hoàn toàn
            self.thread = None
            logging.debug("Đã dừng thread video.")

    def update_fps_display(self, fps):
        """Cập nhật giá trị FPS và hiển thị trên thanh trạng thái."""
        self.current_fps = fps
        self.update_status_bar()    

    def update_resource_usage(self):
        """Cập nhật thông tin sử dụng CPU, RAM, và GPU."""
        # CPU
        self.resource_info["cpu_usage"] = psutil.cpu_percent(interval=None)

        # RAM
        memory = psutil.virtual_memory()
        self.resource_info["ram_usage"] = memory.percent
        self.resource_info["ram_used"] = memory.used / (1024 ** 3)  # GB
        self.resource_info["ram_total"] = memory.total / (1024 ** 3)  # GB

        # GPU
        self.resource_info["gpu_usage"] = 0
        self.resource_info["gpu_memory_used"] = 0
        self.resource_info["gpu_memory_total"] = 0
        if self.gpu_available:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    self.resource_info["gpu_usage"] = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.resource_info["gpu_memory_used"] = mem_info.used / (1024 ** 3)  # GB
                    self.resource_info["gpu_memory_total"] = mem_info.total / (1024 ** 3)  # GB
            except pynvml.NVMLError as e:
                logging.warning(f"Lỗi khi lấy thông tin GPU: {e}")

        self.update_status_bar()
        logging.info(f"Resource update: CPU {self.resource_info['cpu_usage']:.1f}%, "
                     f"RAM {self.resource_info['ram_used']:.1f}/{self.resource_info['ram_total']:.1f} GB, "
                     f"GPU {self.resource_info['gpu_usage']}%")

    def update_status_bar(self):
        """Kết hợp FPS và thông tin tài nguyên để hiển thị trên thanh trạng thái."""
        status_msg = f"FPS: {self.current_fps:.2f} | "
        status_msg += (
            f"CPU: {self.resource_info['cpu_usage']:.1f}% | "
            f"RAM: {self.resource_info['ram_used']:.1f}/{self.resource_info['ram_total']:.1f} GB "
            f"({self.resource_info['ram_usage']:.1f}%)"
        )
        if self.gpu_available:
            status_msg += (
                f" | GPU: {self.resource_info['gpu_usage']}% | "
                f"GPU Mem: {self.resource_info['gpu_memory_used']:.1f}/{self.resource_info['gpu_memory_total']:.1f} GB"
            )
        self.status_bar.showMessage(status_msg)

    def closeEvent(self, event):
        """Dọn dẹp tài nguyên trước khi đóng ứng dụng."""
        logging.info("Đang đóng ứng dụng, dọn dẹp tài nguyên...")
        self.stop_thread()
        
        # TẮT EXECUTOR CỦA FACE PROCESSOR
        if self.face_processor:
            self.face_processor.shutdown()
        
        self.known_students = None
        self.known_students_dict = None
        if self.gpu_available:
            pynvml.nvmlShutdown()  # Tắt pynvml
        self.resource_timer.stop()  # Dừng timer

        gc.collect()
        logging.info("Đã dọn dẹp xong. Tạm biệt!")
        event.accept()

if __name__ == "__main__":
    logging.basicConfig(encoding='utf-8', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("performance.log"),
            logging.StreamHandler()
        ]
    )
    db.create_table()
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())