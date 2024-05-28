import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle


def load_images_from_folder(folder, img_size=(128, 128)):
    images = []
    labels = []
    for breed in os.listdir(folder):
        breed_path = os.path.join(folder, breed)
        if os.path.isdir(breed_path):
            for img in os.listdir(breed_path):
                img_path = os.path.join(breed_path, img)
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = image.resize(img_size)
                    images.append(np.array(image))
                    labels.append(breed)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)


# Test/Training-Split
train_images, train_labels = load_images_from_folder('DataDogs')
test_images, test_labels = load_images_from_folder('DataDogs')

# Label-Encoding für die Labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# CNN-Modell definieren
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Modell kompelieren
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modell fit
model.fit(train_images, train_labels_encoded, epochs=10, validation_data=(test_images, test_labels_encoded))

# das trainierte Modell und den LabelEncoder speichern
model.save('dog_breed_classifier.h5')

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
