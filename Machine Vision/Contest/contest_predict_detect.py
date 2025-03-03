from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)
FOOD = "Sushi"

def load_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return tf.convert_to_tensor(img_array)

model = load_model(f"food_pair_{FOOD}.keras")

dataframe = pd.read_csv(f"data_csv/pair/data_{FOOD}.csv", delimiter=",", header=0)

predictions = []
for index, row in dataframe.iterrows():
    image_1 = load_image(f"datasets/pair/{FOOD}/{row['Image 1']}")
    image_2 = load_image(f"datasets/pair/{FOOD}/{row['Image 2']}")

    prediction = model.predict([tf.expand_dims(image_1, axis=0), tf.expand_dims(image_2, axis=0)])
    
    predictions.append(2 if prediction > 0.5 else 1)

dataframe["Prediction"] = predictions

correct_predictions = dataframe[dataframe["Winner"] == dataframe["Prediction"]].shape[0]
total_predictions = dataframe.shape[0]
accuracy = correct_predictions / total_predictions

dataframe.to_csv("updated_data_with_predictions.csv", index=False)
print(f"Accuracy: {accuracy * 100:.2f}%")
