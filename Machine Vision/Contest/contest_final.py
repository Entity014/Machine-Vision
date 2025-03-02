from keras.models import load_model
import numpy as np
import pandas as pd
import cv2
from tabulate import tabulate

FOLDER_PATH = "datasets/Test Images/"
CSV_PATH = "data_csv/temp/test.csv"
IMAGE_SIZE_CLASS = (128, 128)
IMAGE_SIZE_PAIR = (224, 224)
CLASS_INDICES = {0: "Burger", 1: "Dessert", 2: "Pizza", 3: "Ramen", 4: "Sushi"}

classify_model = load_model("food_classify.keras")


def load_image(image_path, size):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def class_prediction(image):
    predict = classify_model.predict(image)
    predicted_class_index = np.argmax(predict, axis=-1)[0]
    confidence = np.max(predict) * 100  # Convert to percentage
    predicted_class_name = CLASS_INDICES.get(predicted_class_index, "Unknown")
    return predicted_class_name, confidence


def pair_prediction(image1, image2, model):
    predict = model.predict([image1, image2])
    return 2 if predict > 0.5 else 1


dataframe = pd.read_csv(CSV_PATH, delimiter=",", header=0)

pair_models = {}
predictions = []
table_data = []

for _, row in dataframe.iterrows():
    img1_path = f"{FOLDER_PATH}/{row['Image 1']}"
    img2_path = f"{FOLDER_PATH}/{row['Image 2']}"

    image_1 = load_image(img1_path, IMAGE_SIZE_CLASS)
    image_2 = load_image(img2_path, IMAGE_SIZE_CLASS)

    food_1, confidence_1 = class_prediction(image_1)
    food_2, confidence_2 = class_prediction(image_2)

    food_type = food_1 if confidence_1 >= confidence_2 else food_2

    image_1 = load_image(img1_path, IMAGE_SIZE_PAIR)
    image_2 = load_image(img2_path, IMAGE_SIZE_PAIR)

    if food_type not in pair_models:
        pair_models[food_type] = load_model(f"food_pair_{food_type}.keras")
    pair_model = pair_models[food_type]

    winner = pair_prediction(image_1, image_2, pair_model)
    predictions.append(winner)

    table_data.append(
        [
            row["Image 1"],
            f"{food_1} {confidence_1:.2f}%",
            row["Image 2"],
            f"{food_2} {confidence_2:.2f}%",
            f"Image {winner} ({food_type})",
        ]
    )

# dataframe["Prediction"] = predictions
# correct_predictions = dataframe[dataframe["Winner"] == dataframe["Prediction"]].shape[0]
# total_predictions = dataframe.shape[0]
# accuracy = correct_predictions / total_predictions
# print(f"Accuracy: {accuracy * 100:.2f}%")

dataframe["Winner"] = predictions
dataframe.to_csv("updated_data_with_predictions.csv", index=False)

headers = ["Image 1", "Prediction 1", "Image 2", "Prediction 2", "Predicted Winner"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

print("\nPredictions saved successfully!")
