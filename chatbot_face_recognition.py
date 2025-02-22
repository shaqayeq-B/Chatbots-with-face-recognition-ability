import cv2
import threading
import numpy as np
from gtts import gTTS
import os
import time

# تنظیمات مدل‌های تشخیص چهره
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
MOUTH_CASCADE_PATH = 'haarcascade_mouth.xml'  # نیاز به فایل کاسکاد مخصوص دهان

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        self.mouth_cascade = cv2.CascadeClassifier(MOUTH_CASCADE_PATH)
        self.current_emotion = None
        self.last_emotion_time = 0
        self.emotion_cooldown = 10
        self.running = True

    def analyze_facial_features(self, gray_face):
        # تشخیص ویژگی‌های صورت
        features = {
            'mouths': self.mouth_cascade.detectMultiScale(gray_face, 1.8, 20),
            'eyes': self.eye_cascade.detectMultiScale(gray_face, 1.3, 5)
        }
        
        # تحلیل حالت دهان
        mouth_status = 'neutral'
        if len(features['mouths']) > 0:
            (mx, my, mw, mh) = features['mouths'][0]
            mouth_ratio = mh / gray_face.shape[0]
            mouth_status = 'happy' if mouth_ratio > 0.15 else 'sad'

        # تحلیل حالت چشم‌ها
        eye_status = 'open'
        if len(features['eyes']) >= 2:
            avg_eye_height = np.mean([h for (x,y,w,h) in features['eyes']])
            eye_status = 'closed' if avg_eye_height < 15 else 'open'

        # ترکیب ویژگی‌ها برای تشخیص نهایی
        if mouth_status == 'sad' and eye_status == 'closed':
            return 'sad'
        elif mouth_status == 'happy':
            return 'happy'
        return 'neutral'

    def detect_emotions(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x,y,w,h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                self.current_emotion = self.analyze_facial_features(face_roi)

                # نمایش وضعیت تشخیص
                cv2.putText(frame, f"Emotion: {self.current_emotion}", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    def handle_emotion_response(self):
        while self.running:
            if self.current_emotion and (time.time() - self.last_emotion_time) > self.emotion_cooldown:
                if self.current_emotion == 'sad':
                    self.speak("به نظر میاد ناراحت هستی. میخوای در موردش صحبت کنیم؟")
                elif self.current_emotion == 'happy':
                    self.speak("چه حال خوبی دارین امروز! خوشحالم میبینمتون")
                
                self.last_emotion_time = time.time()
            time.sleep(1)

    def speak(self, text):
        tts = gTTS(text=text, lang='fa')
        tts.save("output.mp3")
        os.system("start output.mp3")

if __name__ == "__main__":
    detector = EmotionDetector()
    
    detection_thread = threading.Thread(target=detector.detect_emotions)
    response_thread = threading.Thread(target=detector.handle_emotion_response)

    detection_thread.start()
    response_thread.start()

    detection_thread.join()
    response_thread.join()