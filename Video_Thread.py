import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import config
import logging
import face_processor
from time import time


class VideoThread(QThread):
    # Các tín hiệu không đổi
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_results_signal = pyqtSignal(list)
    finished_signal = pyqtSignal()
    progress_signal = pyqtSignal(int, int)
    error_signal = pyqtSignal(str)
    fps_signal = pyqtSignal(float)

    def __init__(self, input_source, known_students, face_processor, parent=None):
        super().__init__(parent)
        self.input_source = input_source
        self.known_students = known_students
        self._run_flag = True
        self._is_paused = False
        self.cap = None
        self.needs_rotation = False
        self.frame_count = 0
        self.total_frames = 0
        self.face_processor = face_processor
        self.prev_frame = None
        # Kết nối tín hiệu mới này với tín hiệu cũ để không phải sửa gui2.py
        self.processing_in_progress = False # Thêm cờ để tránh gửi quá nhiều yêu cầu
        self.last_seek_time = 0
        self.seek_debounce_interval = 0.5  # 0.5 giây giữa các yêu cầu tua
        # Thêm biến để đo FPS
        self.fps_start_time = time()
        self.fps_frame_count = 0

    def detect_motion(self, current_frame):
        """Kiểm tra xem có chuyển động đáng kể giữa frame hiện tại và frame trước đó."""
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return False
        # Thu nhỏ hình và dùng ảnh xám để giảm chi phí tính toán
        prev_small = cv2.resize(self.prev_frame, (64, 64))
        curr_small = cv2.resize(current_frame, (64, 64))
        diff = cv2.absdiff(cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY), 
                        cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY))
        mean_diff = np.mean(diff)
        self.prev_frame = current_frame.copy()
        return mean_diff > config.MOTION_THRESHOLD
    
    def _read_and_emit_frame(self):
        """Hàm trợ giúp: Đọc 1 frame, xoay nếu cần và gửi tín hiệu đi."""
        if not self.cap or not self.cap.isOpened():
            logging.error("VideoCapture không khả dụng hoặc chưa được khởi tạo.")
            self.error_signal.emit("Không thể truy cập video/camera.")
            self._run_flag = False
            return None
        ret, cv_img = self.cap.read()
        if not ret:
            logging.warning(f"Không thể đọc frame {self.frame_count} từ video.")
            return None
        if self.needs_rotation:
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #logging.debug(f"Đã đọc và gửi frame {self.frame_count}.")
        return cv_img

    def cleanup(self):
        """Dọn dẹp tài nguyên của thread."""
        logging.debug("Dọn dẹp tài nguyên VideoThread.")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def run(self):
        if not isinstance(self.input_source, (int, str)):
            error_msg = "Nguồn đầu vào không hợp lệ."
            logging.error(error_msg)
            self.error_signal.emit(error_msg)
            self.finished_signal.emit()
            return

        # Chỉ định backend cho VideoCapture tăng tốc mở camera
        if isinstance(self.input_source, int):
            self.cap = cv2.VideoCapture(self.input_source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.input_source, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            error_msg = "Không thể mở nguồn video/camera."
            logging.error(error_msg)
            self.error_signal.emit(error_msg)
            self.finished_signal.emit()
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.needs_rotation = video_h > video_w
        #logging.debug(f"Khởi tạo video: {video_w}x{video_h}, total frames: {self.total_frames}")

        last_gui_update = 0
        gui_update_interval = 1.0 / 30  # 30 FPS
        while self._run_flag:
            while self._is_paused and self._run_flag:
                self.msleep(50)
                continue
            
            ret, cv_img = self.cap.read()
            if not ret:
                if isinstance(self.input_source, str):  # Nếu là video file
                    logging.info("Đã phát hết video, tua về đầu và tạm dừng.")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_count = 0
                    self._is_paused = True
                    continue  # Không break, mà tiếp tục vòng lặp với trạng thái pause
                else:
                    logging.info("Camera không còn khung hình, kết thúc thread.")
                    break  # Nếu là camera thì thoát hẳn
            
            current_time = time()
            if current_time - last_gui_update >= gui_update_interval:
                self.change_pixmap_signal.emit(cv_img)
                last_gui_update = current_time

            # Cập nhật FPS
            self.fps_frame_count += 1
            current_time = time()
            elapsed_time = current_time - self.fps_start_time
            if elapsed_time >= 1.0:
                fps = self.fps_frame_count / elapsed_time
                self.fps_signal.emit(fps)
                self.fps_frame_count = 0
                self.fps_start_time = current_time

            # KIỂM TRA ĐIỀU KIỆN XỬ LÝ
            should_process = (self.frame_count % config.SKIP_FRAMES == 0) or self.detect_motion(cv_img)

            # CHỈ GỬI YÊU CẦU NẾU KHÔNG CÓ YÊU CẦU NÀO ĐANG CHỜ
            if should_process and not self.processing_in_progress:
                self.processing_in_progress = True
                # Tạo một bản sao của frame để gửi đi xử lý
                frame_copy = cv_img.copy()

                # Gọi hàm bất đồng bộ mới
                self.face_processor.submit_face_recognition_task(
                    frame_copy, 
                    self.known_students, 
                    self.on_recognition_complete # Một callback để reset cờ
                )
            
            self.progress_signal.emit(self.frame_count, self.total_frames)
            self.frame_count += 1
            self.msleep(10)

        # Dọn dẹp tài nguyên
        self.cleanup()
        self.finished_signal.emit()

    def on_recognition_complete(self, results):
            """Callback được gọi khi xử lý khuôn mặt hoàn tất."""
            self.update_results_signal.emit(results)
            self.processing_in_progress = False # Reset cờ để có thể gửi yêu cầu mới

    def toggle_pause(self):
        """Bật/tắt trạng thái tạm dừng."""
        self._is_paused = not self._is_paused

    def seek_to_frame(self, frame_number):
        """
        Tua đến frame được chỉ định, hiển thị và gửi yêu cầu nhận diện cho frame đó.
        """
        start_time = time()
        if not self.cap or not self.cap.isOpened():
            logging.error("VideoCapture không khả dụng để tua frame.")
            self.error_signal.emit("Không thể tua video: VideoCapture không khả dụng.")
            return

        current_time = time()
        if current_time - self.last_seek_time < self.seek_debounce_interval:
            return  # Bỏ qua nếu tua quá nhanh

        try:
            self.last_seek_time = current_time
            self.frame_count = max(0, min(frame_number, self.total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

            # Đọc frame ngay sau khi tua
            ret, frame = self.cap.read()
            if ret:
                if self.needs_rotation:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # ✨ Hiển thị ngay lên GUI
                self.change_pixmap_signal.emit(frame)

                # ✨ Gửi nhận diện khuôn mặt nếu chưa có tác vụ đang chạy
                if not self.processing_in_progress:
                    self.processing_in_progress = True
                    self.face_processor.submit_face_recognition_task(
                        frame.copy(), self.known_students, self.on_recognition_complete
                    )

                # ✨ Cập nhật tiến độ
                self.progress_signal.emit(self.frame_count, self.total_frames)
            else:
                logging.warning(f"Không thể đọc frame {self.frame_count} sau khi tua.")

            logging.debug(f"Tua đến frame: {self.frame_count}")

        except Exception as e:
            logging.error(f"Lỗi khi tua đến frame {frame_number}: {str(e)}")
            self.error_signal.emit(f"Lỗi khi tua đến frame: {str(e)}")
            self.processing_in_progress = False

        finally:
            end_time = time()
            response_time = (end_time - start_time) * 1000  # Chuyển sang ms
            logging.info(f"Thời gian phản hồi khi tua video: {response_time:.2f} ms")


    def stop(self):
        self._run_flag = False