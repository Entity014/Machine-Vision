from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

IM_SIZE = 128
BATCH_SIZE = 50
MAX_EPOCH = 100

# Create Model
# base_model = EfficientNetB0(
#     weights="imagenet",
#     include_top=False,  # ? include_top=False : ไม่เอา layer สุดท้าย
#     input_shape=(IM_SIZE, IM_SIZE, 3),
# )
# base_model = ResNet50(
#     weights="imagenet", include_top=False, input_shape=(IM_SIZE, IM_SIZE, 3)
# )
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(IM_SIZE, IM_SIZE, 3)
)

encoder_feature_map = base_model.output
avg_feature_map = GlobalAveragePooling2D()(
    encoder_feature_map
)  # ? GlobalAveragePooling2D : ทำให้เป็น 1D โดยเฉลี่ยค่าในแต่ละ layer
dense = Dense(64, activation="relu")(avg_feature_map)
output = Dense(5, activation="softmax")(dense)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen.flow_from_directory(
    "datasets/class/train",
    shuffle=True,
    target_size=(IM_SIZE, IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
)

validation_generator = datagen.flow_from_directory(
    "datasets/class/val",
    shuffle=False,
    target_size=(IM_SIZE, IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
)

test_generator = datagen.flow_from_directory(
    "datasets/class/test",
    shuffle=False,
    target_size=(IM_SIZE, IM_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
)


# Train Model
checkpoint = ModelCheckpoint(
    "food_classify.keras",
    verbose=1,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
)
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)


h = model.fit(
    train_generator,
    epochs=MAX_EPOCH,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint, early_stopping, reduce_lr],
)

plt.plot(h.history["accuracy"])
plt.plot(h.history["val_accuracy"])
plt.legend(["train", "val"])


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
