import cv2
import mediapipe as mp
import time

# Hand detection class
class handDetector():
    def __init__(self ):
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # Save results to self.results
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, image = cap.read()
        if not success:
            break  # Exit the loop if the frame was not read successfully
        image = detector.findhands(image)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop when 'q' is pressed
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
