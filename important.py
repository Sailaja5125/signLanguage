# primary 
import cv2
import mediapipe as mp
import time
from deepface import DeepFace
# important code and most accurate
# Creating video object
cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils

# Emoji mapping for emotions with image paths
emotion_emojis = {
    'angry': 'emojis/angry.png',
    'disgust': 'emojis/nauseated.png',
    'fear': 'emojis/fearful.png',
    'happy': 'emojis/grinning.png',
    'sad': 'emojis/pensive.png',
    'surprise': 'emojis/surprised.png',
    'neutral': 'emojis/neutral.png'
}

pTime = 0
cTime = 0

while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(image, bbox, (255, 0, 0), 2)

            # Ensure the bounding box coordinates are within the image dimensions
            x_min = max(0, bbox[0])
            y_min = max(0, bbox[1])
            x_max = min(iw, bbox[0] + bbox[2])
            y_max = min(ih, bbox[1] + bbox[3])

            # Crop the face region safely
            face = image[y_min:y_max, x_min:x_max]

            # Resize the face region to a standard size expected by DeepFace
            face_resized = cv2.resize(face, (48, 48))
            # Use DeepFace to analyze emotions
            try:
                result = DeepFace.analyze(face_resized, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list) and len(result) > 0:
                    emotion = result[0]['dominant_emotion']
                    emoji_path = emotion_emojis.get(emotion)  # Get emoji path

                    # Load and resize emoji image to fit on screen if needed
                    emoji_image = cv2.imread(emoji_path)
                    emoji_image_resized = cv2.resize(emoji_image, (30, 30))  # Resize as needed

                    # Place emoji on the image at a specified location
                    image[y_min:y_min + emoji_image_resized.shape[0], x_min:x_min + emoji_image_resized.shape[1]] = emoji_image_resized
                    
                    # Display emotion text above the bounding box
                    cv2.putText(image, emotion, (bbox[0], bbox[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.putText(image, 'No emotion detected', (bbox[0], bbox[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception as e:
                cv2.putText(image, 'Analysis error', (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"Analysis error: {e}")
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 3,
                (255, 0, 255), 3)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()