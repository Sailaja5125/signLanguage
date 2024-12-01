import os
import cv2
import mediapipe as mp
import time
from deepface import DeepFace

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (255, 0, 0)
FONT_THICKNESS = 2
EMOJI_SIZE = (40, 40)
FACE_SIZE = (58, 58)
FPS_POSITION = (10, 70)
FPS_FONT_SCALE = 3
FPS_COLOR = (255, 0, 255)

# Emoji mapping for emotions with image paths
emotion_emojis = {
    'angry': 'emojis/angry.png',
    'disgust': 'emojis/disgust.png',
    'fear': 'emojis/fear.png',
    'happy': 'emojis/grinning.png',
    'sad': 'emojis/pensive.png',
    'surprise': 'emojis/surprised.png',
    'neutral': 'emojis/neutral.png'
}

class EmotionAnalyzer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_detection = mp.solutions.face_detection.FaceDetection()
        self.pTime = 0

    # analyze the emotions
    def analyze_emotion(self, face):
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list) and len(result) > 0:
                return result[0]['dominant_emotion']
        except Exception as e:
            print(f"Analysis error: {e}")
        return None
    # draw emoji and text 
    def draw_emoji_and_text(self, image, bbox, emotion):
        if emotion:
            emoji_path = emotion_emojis.get(emotion)
            if emoji_path:
                emoji_image = cv2.imread(emoji_path)
                emoji_image_resized = cv2.resize(emoji_image, EMOJI_SIZE)
                x_min, y_min = bbox[0], bbox[1]
                image[y_min:y_min + emoji_image_resized.shape[0], x_min:x_min + emoji_image_resized.shape[1]] = emoji_image_resized
                cv2.putText(image, emotion, (bbox[0], bbox[1] - 10), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        else:
            cv2.putText(image, 'No emotion detected', (bbox[0], bbox[1] - 10), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
 
     # process the video frame 
    def process_frame(self, image):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(imgRGB)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, bbox, (255, 0, 0), 2)

                x_min = max(0, bbox[0])
                y_min = max(0, bbox[1])
                x_max = min(iw, bbox[0] + bbox[2])
                y_max = min(ih, bbox[1] + bbox[3])
                face = image[y_min:y_max, x_min:x_max]
                face_resized = cv2.resize(face, FACE_SIZE)

                emotion = self.analyze_emotion(face_resized)
                self.draw_emoji_and_text(image, bbox, emotion)

        return image
    # display the fps
    def display_fps(self, image):
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(image, str(int(fps)), FPS_POSITION, FONT, FPS_FONT_SCALE, FPS_COLOR, FONT_THICKNESS)

    def run(self):
        try:
            while True:
                success, image = self.cap.read()
                if not success:
                    print("Failed to capture image")
                    break

                image = self.process_frame(image)
                self.display_fps(image)
                cv2.imshow("Image", image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = EmotionAnalyzer()
    analyzer.run()