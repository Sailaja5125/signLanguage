import os
os.environ['TFF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import numpy as np



physical_devices = tf.config.list_physical_devices('GPU')

# Define paths to your dataset directories
train_dir = './Train'
test_dir = './Test'

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2
)

# Just normalization for testing/validation
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and iterate training dataset with target size of (32, 32)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),  # Change to (32, 32)
    batch_size=32,
    class_mode='sparse',
    color_mode='rgb'
)

# Load and iterate test dataset with target size of (32, 32)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(32, 32),  # Change to (32, 32)
    batch_size=32,
    class_mode='sparse',
    color_mode='rgb'
)

# Build the model using VGG16 as a base model with input shape (32, 32, 3)
base_model = tf.keras.applications.VGG16(input_shape=(32, 32, 3),
                                         include_top=False,
                                         weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Create a new model on top of the base model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer matches number of classes
])

# Compile the model with a lower learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=50, validation_data=test_generator)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"Test accuracy: {test_acc}")

# Make predictions on the test dataset (optional)
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)  # Get class indices of predictions

# Print predicted classes for the first test image in the batch (for demonstration)
print(f"Predicted class indices: {predicted_classes[:10]}")  # Display first 10 predictions

# Save the trained model
model.save('model.h5')