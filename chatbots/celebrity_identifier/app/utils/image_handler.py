import cv2
import numpy as np
from io import BytesIO

class ProcessImage:

    def __init__(self):
        self.method = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def process_image(self, image_file):
        temp_file = BytesIO()
        image_file.save(temp_file)

        image_bytes = temp_file.getvalue()
        img_arr = np.frombuffer(image_bytes, dtype=np.uint8)

        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        face = self.method.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        if not len(face):
            return image_bytes, None

        # for (x, y, w, h) in face:
        face = max(face, key=lambda x: x[-1]*x[-2])
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        _, buffer = cv2.imencode('.jpg', image)
        
        return buffer.tobytes(), face

    