# Colab Version: https://colab.research.google.com/drive/1k3PclpSOtHmnlWnv7P6zAG-KZw6nxZV-?usp=sharing

from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

IM_SIZE = 128
BATCH_SIZE = 50
MAX_EPOCH = 100


datagen = ImageDataGenerator(rescale=1.0 / 255)


test_generator = datagen.flow_from_directory(
    "datasets/class/test",
    shuffle=False,
    target_size=(IM_SIZE, IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
)


# Test Model

model = load_model("food_classify.keras")
score = model.evaluate(test_generator, steps=len(test_generator))
print("score (cross_entropy, accuracy):\n", score)


test_generator.reset()
predict = model.predict(test_generator, steps=len(test_generator))
print("confidence:\n", predict)

predict_class_idx = np.argmax(predict, axis=-1)
print("predicted class index:\n", predict_class_idx)

mapping = dict((v, k) for k, v in test_generator.class_indices.items())
predict_class_name = [mapping[x] for x in predict_class_idx]
print("predicted class name:\n", predict_class_name)

cm = confusion_matrix(test_generator.classes, np.argmax(predict, axis=-1))
print("Confusion Matrix:\n", cm)

plt.show()
