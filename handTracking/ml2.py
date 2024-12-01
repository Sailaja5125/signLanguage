import cv2
import mediapipe as mp
import time

# Create a video object
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Detection
mpFaceDetection = mp.solutions.face_detection
face_detection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Initialize variables for FPS calculation
pTime = 0
cTime = 0

while True:
    success, image = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Convert the image to RGB format
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect faces
    results = face_detection.process(imgRGB)

    # Draw bounding boxes around detected faces
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x_min = int(bboxC.xmin * iw)
            y_min = int(bboxC.ymin * ih)
            width = int(bboxC.width * iw)
            height = int(bboxC.height * ih)

            # Draw rectangle around the face
            cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (255, 0, 0), 2)

            # Optionally draw landmarks (not available in face detection)
            # mpDraw.draw_landmarks(image, detection)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the image
    cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Show the image with detections
    cv2.imshow("Face Detection", image)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()