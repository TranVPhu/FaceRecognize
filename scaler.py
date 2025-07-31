# scaler.py

import cv2

class FixedScaler:
    def __init__(self, target_width=640):
        self.target_width = target_width
        self.scale_ratio = 1.0

    def resize_image(self, img):
        """Resize ảnh theo target_width, và ghi lại scale_ratio."""
        h, w = img.shape[:2]
        self.scale_ratio = self.target_width / w
        new_h = int(h * self.scale_ratio)
        resized = cv2.resize(img, (self.target_width, new_h))
        return resized

    def scale_location(self, location):
        """Scale bounding box từ ảnh đã resize sang ảnh gốc."""
        top, right, bottom, left = location
        s = 1.0 / self.scale_ratio
        return (
            int(top * s),
            int(right * s),
            int(bottom * s),
            int(left * s)
        )

    def scale_location_forward(self, location):
        """Scale bounding box từ ảnh gốc sang ảnh đã resize."""
        top, right, bottom, left = location
        s = self.scale_ratio
        return (
            int(top * s),
            int(right * s),
            int(bottom * s),
            int(left * s)
        )

    def unscale_point(self, x, y):
        """Click từ ảnh nhỏ về ảnh gốc"""
        s = 1.0 / self.scale_ratio
        return int(x * s), int(y * s)
