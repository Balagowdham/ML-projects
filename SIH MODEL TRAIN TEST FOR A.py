import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
img_width, img_height = 1920, 1080  # Reduced image size
batch_size = 16  # Reduced batch size
num_classes = 3  # Change based on your classes

# Prepare data with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load training data
training_set = train_datagen.flow_from_directory(
    'saved_images',  # Directory with training images
    target_size=(img_height, img_width),  # Resizing images
    batch_size=batch_size,
    class_mode='sparse'  # Use 'categorical' if using one-hot encoding
)

# Calculate class weights to handle class imbalance
labels = training_set.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(labels), 
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))

# Define a simpler CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(64, activation='relu'),  # Reduced number of units
    Dropout(0.5),
    
    Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(
    training_set,
    epochs=5,  # Adjust epochs as needed
    class_weight=class_weights_dict,
    verbose=1
)

# Save the trained model
model.save('hand_sign_recognition_model.h5')
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
final_accuracy = history.history['accuracy'][-1]
print(f"Final Training Accuracy: {final_accuracy:.4f}")
