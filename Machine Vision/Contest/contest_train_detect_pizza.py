from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Concatenate,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import pandas as pd
import matplotlib.pyplot as plt


BATCH_SIZE = 14
MAX_EPOCH = 100
IMAGE_SIZE = (224, 224)
FOOD = "Pizza"

# Base Model (MobileNetV2)
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

encoder_feature_map = base_model.output
avg_feature_map = GlobalAveragePooling2D()(encoder_feature_map)
encoder = Model(inputs=base_model.input, outputs=avg_feature_map)

# Siamese Network Model
input_1 = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
feature_1 = encoder(input_1)

input_2 = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
feature_2 = encoder(input_2)

concat = Concatenate()([feature_1, feature_2])

dense1 = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(concat)
batch_norm1 = BatchNormalization()(dense1)
dropout1 = Dropout(0.4)(batch_norm1)
output = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.001))(dropout1)

siamese_model = Model(inputs=[input_1, input_2], outputs=output)

siamese_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Data generator with augmentation
def myGenerator(type, food=FOOD):
    # Read CSV data
    dataframe = pd.read_csv(f"data_csv/pair/data_{food}.csv", delimiter=",", header=0)
    dataframe["Winner"] = dataframe["Winner"].apply(lambda x: 0 if x == 1 else 1)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,  # ปรับเพิ่มการหมุน
        width_shift_range=0.3,  # ปรับการขยับ
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,  # ปรับขนาดเพิ่ม
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
    )

    datagen_noarg = ImageDataGenerator(rescale=1.0 / 255)

    # Select generators for training and validation
    if type == "train":
        input_generator_1 = datagen.flow_from_dataframe(
            dataframe=dataframe.loc[0:int(dataframe.shape[0] * 0.8) - 1],
            directory=f"datasets/pair/{food}",
            x_col="Image 1",
            y_col="Winner",
            class_mode="other",
            color_mode="rgb",
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=1,
        )

        input_generator_2 = datagen.flow_from_dataframe(
            dataframe=dataframe.loc[0:int(dataframe.shape[0] * 0.8) - 1],
            directory=f"datasets/pair/{food}",
            x_col="Image 2",
            y_col="Winner",
            class_mode="other",
            color_mode="rgb",
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=1,
        )
    else:  # For validation
        input_generator_1 = datagen_noarg.flow_from_dataframe(
            dataframe=dataframe.loc[int(dataframe.shape[0] * 0.8):int(dataframe.shape[0] * 0.9) - 1],
            directory=f"datasets/pair/{food}",
            x_col="Image 1",
            y_col="Winner",
            class_mode="other",
            color_mode="rgb",
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=1,
        )

        input_generator_2 = datagen_noarg.flow_from_dataframe(
            dataframe=dataframe.loc[int(dataframe.shape[0] * 0.8):int(dataframe.shape[0] * 0.9) - 1],
            directory=f"datasets/pair/{food}",
            x_col="Image 2",
            y_col="Winner",
            class_mode="other",
            color_mode="rgb",
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=1,
        )

    while True:
        in_batch_1 = next(input_generator_1)
        in_batch_2 = next(input_generator_2)
        yield (in_batch_1[0], in_batch_2[0]), in_batch_1[1]


# Define Callbacks
checkpoint = ModelCheckpoint(f"food_pair_{FOOD}.keras", verbose=1, monitor="val_accuracy", save_best_only=True, mode="max")
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

dataframe = pd.read_csv(f"data_csv/pair/data_{FOOD}.csv", delimiter=",", header=0)

# Train Model
h = siamese_model.fit(
    myGenerator("train"),
    steps_per_epoch=int((dataframe.shape[0] * 0.8 // BATCH_SIZE)),
    epochs=MAX_EPOCH,
    validation_data=myGenerator("validation"),
    validation_steps=int((dataframe.shape[0] * 0.1 // BATCH_SIZE)),
    callbacks=[checkpoint, early_stopping, reduce_lr],
)

# Plotting the results
plt.plot(h.history["accuracy"])
plt.plot(h.history["val_accuracy"])
plt.legend(["train", "val"])
plt.show()
