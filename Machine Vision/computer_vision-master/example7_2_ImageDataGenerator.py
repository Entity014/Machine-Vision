# For Goole Colab Version
# https://colab.research.google.com/drive/138XnTYRSe4HIg_XELX-RixCVHuxDQHao?usp=share_link

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

IM_SIZE = 64

# Create model
input = Input(shape=(IM_SIZE, IM_SIZE, 3))
conv1 = Conv2D(8, 3, activation="relu")(input)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(8, 3, activation="relu")(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
flat = Flatten()(pool2)
hidden = Dense(16, activation="relu")(flat)
output = Dense(12, activation="softmax")(hidden)
model = Model(inputs=input, outputs=output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()


# Create generator (download dataset form https://drive.google.com/drive/folders/1jiwbiSbEMQkVGg3Oq2TZI2M9CPm0cJwC?usp=sharing)
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen.flow_from_directory(
    "animalfaces/train",
    shuffle=True,
    target_size=(IM_SIZE, IM_SIZE),
    batch_size=50,
    color_mode="rgb",
    class_mode="categorical",
)

validation_generator = datagen.flow_from_directory(
    "animalfaces/validation",
    shuffle=False,
    target_size=(IM_SIZE, IM_SIZE),
    batch_size=50,
    color_mode="rgb",
    class_mode="categorical",
)

test_generator = datagen.flow_from_directory(
    "animalfaces/test",
    shuffle=False,  # ? ไม่ต้อง shuffle เพราะเราจะงง
    target_size=(IM_SIZE, IM_SIZE),
    batch_size=50,
    color_mode="rgb",
    class_mode="categorical",
)


# Train Model
checkpoint = ModelCheckpoint(
    "animalfaces.h5", verbose=1, monitor="val_accuracy", save_best_only=True, mode="max"
)

h = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=len(
        train_generator
    ),  # ? ครบ 1 epoch จะ validation data กี่ครั้ง ; อยากให้นานๆ validate ที ให้คูณ 2 ค่านี้ไป
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint],
)

plt.plot(h.history["accuracy"])
plt.plot(h.history["val_accuracy"])
plt.legend(["train", "val"])


# Test Model
model = load_model("animalfaces.h5")
score = model.evaluate(test_generator, steps=len(test_generator))
print("score (cross_entropy, accuracy):\n", score)


test_generator.reset()  # ? ต้อง reset ก่อนทุกครั้งที่จะใช้ generator ใหม่
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
