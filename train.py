from data_loader import load_darpa
from neurallog.models.transformers import transformer_classifer
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

embed_dim = 768  # Embedding size for each token

# OVERLY COMPLICATED MODEL...
# num_heads = 12  # Number of attention heads
# ff_dim = 2048  # Hidden layer size in feed forward network inside transformer

# Tested with something stupid like having "MALICIOUS" makes up the entire positive case, and "BENIGN" for negative... and it still didnt work with the above parameters
# Worked with like num_head = 1 and ff_dim = 32 tho...

num_heads = 6  # Number of attention heads
ff_dim = 256  # Hidden layer size in feed forward network inside transformer
max_len = 200

(x_train, y_train), (x_test, y_test) = load_darpa(npz_file="data-secbert.npz")
x_train = np.array([np.array(sublist) for sublist in x_train])
# some_set = {}
# for row in x_train:
#     original = row[0]
#     if original.sum() not in some_set:
#         some_set[original.sum()] = 1
#     some_set[original.sum()] +=1
# x_train = np.array([np.array(sublist) for sublist in x_train])
# cum = np.sum(x_train, axis=2)
# print(cum[y_train == 0])
# y_train = keras.utils.to_categorical(y_train)

def create_model(embed_dim=768, ff_dim=256, max_len=200, num_heads=6, lr=5e-6):
    model = transformer_classifer(embed_dim, ff_dim, max_len, num_heads)
    model.compile(optimizer=Adam(learning_rate=lr), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0, ff_dim=64, lr=5e-6, num_heads=4)

advanced_grid = {
    'ff_dim': [64, 128, 256, 512],
    'lr': [5e-6, 1e-5, 2e-5],
    'num_heads': [2, 4, 8]
}

# Best: 0.512500 using {'ff_dim': 4, 'lr': 5e-06} with normal BERT
# Best: 0.550000 using {'ff_dim': 4, 'lr': 1e-06} with secBERT
param_grid = {
    'ff_dim': [4, 8, 16],
    'lr': [1e-6, 5e-6, 1e-5]
}

simple_grid = {}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and corresponding accuracy
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# batch_size = 16
# epochs = 10

# history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
