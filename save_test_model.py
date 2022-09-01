from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the digits dataset
digits = datasets.load_digits()

encoder = preprocessing.LabelBinarizer()
encoder.fit(digits.target)
target = encoder.transform(digits.target)
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, target, test_size=0.25, random_state=0
)


print(f"x_train: {x_train.shape}")
print(f"x_test: {x_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

model = Sequential()
model.add(Dense(64, activation="relu", input_dim=64))
model.add(Dense(10, activation="softmax"))
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=100, batch_size=32)
