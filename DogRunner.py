import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

# Lade das trainierte Modell und den LabelEncoder
model = load_model('/Users/erwinlang/PycharmProjects/DogID/dog_breed_classifier.h5')
with open('/Users/erwinlang/PycharmProjects/DogID/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


def predict_breed(image, model, label_encoder):
    """
    Vorhersage der Hunderasse basierend auf dem Bild.

    :param image: Das Bild des Hundes als numpy Array
    :param model: Das geladene Keras Modell
    :param label_encoder: Der LabelEncoder für die Hunderassen
    :return: Die vorhergesagte Hunderasse
    """
    image = image.resize((128, 128))  # Ändere die Bildgröße entsprechend der Modellanforderungen
    image = np.expand_dims(np.array(image), axis=0)  # Füge Batch-Dimension hinzu
    predictions = model.predict(image)
    predicted_breed = label_encoder.inverse_transform([np.argmax(predictions)])
    return predicted_breed[0]


# Streamlit App Titel
st.title("DogID - Finde die Rasse eines Hundes heraus!")

# Bild hochladen
uploaded_file = st.file_uploader("Füge hier deinen Freund auf vier Beinen ein", type="jpg")

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption='Dein hochgeladenes Bild.', use_column_width=True)

    # Vorhersage treffen
    breed = predict_breed(image, model, label_encoder)
    st.write(f'Toll - Dieser Hund ist ein {breed}!')