from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

tf.keras.backend.clear_session()

# Generate some random data for demonstration
np.random.seed(42)
X_train = np.random.rand(100, 10)  # 100 samples with 10 features each
y_train = np.random.randint(2, size=(100,))  # Binary labels (0 or 1)

# Create a random Keras model
model = Sequential()
model.add(Dense(units=64, input_dim=10, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
