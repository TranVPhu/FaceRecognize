# face_processor.py
"""
Module 1 & 3: Tiền xử lý ảnh, phát hiện và nhận diện khuôn mặt bằng InsightFace (ArcFace).
"""
import cv2
import numpy as np
import insightface
import gc
import math
import logging
import faiss
from config import RESIZE_FACTOR, RECOGNITION_TOLERANCE, DET_SIZE, MAX_WORKERS
from concurrent.futures import ThreadPoolExecutor

class FaceProcessor:
    def __init__(self, faiss_index, id_mapping):
        # Khởi tạo model ArcFace chỉ 1 lần
        try:
            # Thử khởi tạo với GPU trước
            self.model = insightface.app.FaceAnalysis(name='buffalo_sc', providers=['CUDAExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=DET_SIZE)
            logging.info("Đã khởi tạo FaceProcessor với CUDAExecutionProvider.")
        except Exception as e:
            logging.warning(f"Không tìm thấy GPU hoặc lỗi khi khởi tạo CUDA: {e}")
            # Fallback sang CPU
            self.model = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=-1, det_size=DET_SIZE)
            logging.info("Khởi tạo FaceProcessor với CPUExecutionProvider.")
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.faiss_index = faiss_index         
        self.id_mapping = id_mapping 
        print(f"[FaceProcessor] Khởi tạo với {MAX_WORKERS} luồng xử lý.")

    def process_frame_for_faces(self, frame):
        """
        Tiền xử lý một khung hình và phát hiện các khuôn mặt.
        Trả về: vị trí các khuôn mặt và mã hóa của chúng.
        """
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
        faces = self.model.get(small_frame)
        face_locations = []
        face_embeddings = []

        for face in faces:
            # Lấy vị trí khuôn mặt (top, right, bottom, left)
            box = face.bbox.astype(int)
            left, top, right, bottom = box[0], box[1], box[2], box[3]
            face_locations.append((top, right, bottom, left))
            # Lấy embedding (vector đặc trưng)
            face_embeddings.append(face.embedding.astype(float))

        return face_locations, face_embeddings

    def identify_faces(self, face_embeddings, known_students): 
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            num_faces = len(face_embeddings)
            return ["Người lạ"] * num_faces, [None] * num_faces, [0.0] * num_faces

        # Chuyển đổi embedding đầu vào thành mảng numpy và chuẩn hóa
        query_embeddings = np.array(face_embeddings).astype('float32')
        faiss.normalize_L2(query_embeddings)

        # Tìm kiếm 1 vector gần nhất (k=1)
        # D là khoảng cách (L2 distance), I là chỉ số (index) của vector trong file Faiss
        k = 1
        distances, indices = self.faiss_index.search(query_embeddings, k)

        identified_ids = []
        identified_names = []
        similarity_scores = []
        
        # Ngưỡng nhận diện cần được chuyển từ Cosine Similarity sang L2 Distance
        # D^2 = 2 - 2 * S  =>  D = sqrt(2 * (1 - S))
        # RECOGNITION_TOLERANCE cũ là khoảng cách (0.6), nhưng S là độ tương đồng.
        # Giả sử ngưỡng tương đồng là (1 - 0.6) = 0.4
        # Thì ngưỡng khoảng cách là sqrt(2 * (1 - 0.4)) = sqrt(1.2) ~= 1.095
        # Bạn cần tinh chỉnh lại ngưỡng này cho phù hợp
        # InsightFace thường dùng ngưỡng cosine là 0.5, tương ứng L2 là 1.0
        # RECOGNITION_TOLERANCE biểu thị ngưỡng khoảng cách cho phép (1 - S)
        # Ví dụ: tolerance 0.6 -> S = 0.4 -> ngưỡng L2 ≈ sqrt(1.2) ≈ 1.095
        DISTANCE_THRESHOLD = math.sqrt(2 * RECOGNITION_TOLERANCE) 

        # known_students giờ được dùng để tra cứu tên từ ID
        known_students_dict = {s["id"]: s for s in known_students}

        for i in range(len(query_embeddings)):
            faiss_index = indices[i][0]
            distance = distances[i][0]

            if distance < DISTANCE_THRESHOLD:
                # Lấy student_id từ file ánh xạ dựa vào chỉ số của Faiss
                student_id = self.id_mapping[faiss_index]
                # Tra cứu thông tin từ dict
                student_info = known_students_dict.get(student_id)
                if student_info:
                    name = student_info["name"]
                else:
                    name = "Không rõ" # Trường hợp ID có trong Faiss nhưng không có trong dict
                    student_id = None
            else:
                student_id = None
                name = "Người lạ"
            
            # Tính điểm tương đồng từ khoảng cách để hiển thị nếu cần
            score = 1 - (distance**2) / 2
            
            identified_ids.append(student_id)
            identified_names.append(name)
            similarity_scores.append(score)
        
        return identified_names, identified_ids, similarity_scores

    def submit_face_recognition_task(self, frame, known_students, callback):
        """Gửi tác vụ nhận diện vào a thread pool và gọi callback khi hoàn thành."""
        future = self.executor.submit(self._recognize_in_background, frame, known_students)
        future.add_done_callback(lambda f: callback(f.result()))

    def _recognize_in_background(self, frame, known_students):
        """Hàm này sẽ chạy trong một luồng riêng của ThreadPoolExecutor."""
        try:
            locations, encodings = self.process_frame_for_faces(frame)
            frame = None
            if not encodings:
                return [] # Trả về danh sách rỗng nếu không có khuôn mặt

            names, ids, _ = self.identify_faces(encodings, known_students)
            results = [{"name": n, "id": i, "location": l} for n, i, l in zip(names, ids, locations)]
            return results
        except Exception as e:
            # Nên có logging ở đây                                                                                      
            print(f"Lỗi trong luồng xử lý khuôn mặt: {e}")
            return []

    def get_single_face_encoding(self, image_to_process, known_students):
        if image_to_process is None:
            print("[Lỗi] Không có ảnh để xử lý.")
            return None

        # Sử dụng phương thức của chính đối tượng này
        locations, encodings = self.process_frame_for_faces(image_to_process)

        if len(encodings) == 0:
            print("[Lỗi] Không tìm thấy khuôn mặt nào trong ảnh.")
            return None
        if len(encodings) > 1:
            print("[Lỗi] Phát hiện nhiều hơn một khuôn mặt. Vui lòng chỉ có một người trong ảnh.")
            return None
        
        names, ids, _ = self.identify_faces(encodings, known_students)
        results = [{"name": n, "id": i, "location": l} for n, i, l in zip(names, ids, locations)]

        return results, encodings[0]
    
    def clear_cache(self):
        """Dọn dẹp bộ nhớ cache (nếu có) và gọi garbage collector."""
        gc.collect()
        if hasattr(self.model, 'clear'):
            self.model.clear()
        print("Cache cleared.")

    def shutdown(self):
        """Tắt ThreadPoolExecutor và giải phóng tài nguyên."""
        if hasattr(self, 'executor') and self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        if hasattr(self, 'model'):
            self.model = None