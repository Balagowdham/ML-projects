import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(1080, 1920, 3)),  # 1080p input shape
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # 26 classes for A-Z
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the data
train_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('saved_images',
                                                 target_size=(1080, 1920),  # 1080p images
                                                 batch_size=32,
                                                 class_mode='categorical')

# Train the model
model.fit(training_set, epochs=10)

# Save the trained model
model.save('sign_language_model_1080p.h5')
