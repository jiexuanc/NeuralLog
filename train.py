from data_loader import load_darpa
from neurallog.models.transformers import transformer_classifer
from tensorflow import keras
import numpy as np

embed_dim = 768  # Embedding size for each token
num_heads = 12  # Number of attention heads
ff_dim = 2048  # Hidden layer size in feed forward network inside transformer
max_len = 200

(x_train, y_train), (x_test, y_test) = load_darpa(npz_file="data-bert.npz")
x_train = np.array([np.array(sublist) for sublist in x_train])
print(x_train.shape)

model = transformer_classifer(embed_dim, ff_dim, max_len, num_heads)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

batch_size = 16
epochs = 10

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
