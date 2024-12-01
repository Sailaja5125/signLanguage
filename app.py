from flask import Flask, request, jsonify
from flask_cors import CORS , cross_origin
import cv2
import numpy as np
from ml import EmotionAnalyzer

app = Flask(__name__)
CORS(app)
cross_origin("*")
emotion_analyzer = EmotionAnalyzer()

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    # Get the image from the request
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Process the image and get the emotion
    processed_image = emotion_analyzer.process_frame(image)
    emotion = emotion_analyzer.analyze_emotion(processed_image)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run()
