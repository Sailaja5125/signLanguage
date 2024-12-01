import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model.h5')  # Ensure this path is correct

# Function to predict a single image
def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(32, 32))  # Load image with target size
    img_array = img_to_array(img) / 255.0  # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class index and corresponding emotion label
    predicted_class_index = np.argmax(predictions)    
    # Map class index to class label (adjust based on your dataset)
    class_labels = ['anger','fear','joy','Natural','sadness','surprise']  # Adjust based on your classes
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label

# Example usage of predict_image function:
image_path = 'kdh.jpeg'  # Replace with your image path
predicted_emotion = predict_image(image_path)
print(f"The predicted emotion is: {predicted_emotion}")