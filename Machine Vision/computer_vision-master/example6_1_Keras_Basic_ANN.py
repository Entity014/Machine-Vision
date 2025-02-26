# For Google Colab Version
# https://colab.research.google.com/drive/1NNMXzedIkMiDH9GkiK6Tzp9HB9mjb0bl?usp=share_link

from keras.models import Model
from keras.layers import Input, Dense
import numpy as np

# Create model
input = Input(shape=(3,))  # ? Input Layer
# ? Hidden Layer ; Dense(จำนวน Node, activation Function)(Layer ที่ต้องการเชื่อม)
hidden = Dense(4, activation="tanh")(input)
# ? Output Layer
output = Dense(1, activation="sigmoid")(hidden)  # Binary Classification
model = Model(inputs=input, outputs=output)

# ? loss : optimizer จะใช้ , metrics : เราจะใช้
model.compile(
    optimizer="sgd",  # ? Modified Gradient Descent
    loss="binary_crossentropy",  # ? สำหรับ Binary Classification
    metrics=["accuracy"],
)

model.summary()

# Train model (N, M)
# ? Row : จำนวนข้อมูล , Column : จำนวน Feature
x_train = np.asarray(
    [
        [1, 0, 1],
        [4, 2, 0],
        [-4, 0, 1],
        [1, 2, 3],
        [-1, -2, 3],
        [0, -1, 3],
        [1, 0, 0],
        [-1, 0, -2],
        [4, -2, 7],
        [-1, -1, 4],
    ]
)

y_train = np.asarray([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])

model.fit(
    x_train, y_train, epochs=100, batch_size=10
)  # ? batch_size : จำนวนข้อมูลที่ใช้ update weights ในแต่ละครั้ง

# Test model
x_test = np.asarray([[1, 5, 1], [5, -1, 3], [-5, 0, 1], [0, 0, 0]])

y_test = np.asarray([1, 1, 0, 0])

y_pred = model.predict(x_test)

print("y_pred")
print(y_pred)

# ? loss, metrics
score = model.evaluate(x_test, y_test)

print("score")
print(score)
