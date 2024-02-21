from data_loader import load_darpa
from neurallog.models.transformers import transformer_classifer
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping
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

num_heads = 4  # Number of attention heads
ff_dim = 2048  # Hidden layer size in feed forward network inside transformer
max_len = 50

(x_train, y_train), (x_test, y_test) = load_darpa(npz_file="data-bert-flow-cls.npz")
x_train = np.array([np.array(sublist) for sublist in x_train])
x_test = np.array([np.array(sublist) for sublist in x_test])
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

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, mode="max")

def create_model(embed_dim=768, ff_dim=2048, max_len=50, num_heads=12, lr=3e-4):
    model = transformer_classifer(embed_dim, ff_dim, max_len, num_heads)
    model.compile(optimizer=Adam(learning_rate=lr), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

# model = create_model()

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=16, ff_dim=ff_dim, lr=3e-4, num_heads=num_heads, max_len=max_len, callbacks=[early_stopping])

advanced_grid = {
    'ff_dim': [64, 128, 256, 512],
    'lr': [5e-6, 1e-5, 2e-5],
    'num_heads': [2, 4, 8]
}

# Best: 0.512500 using {'ff_dim': 4, 'lr': 5e-06} with normal BERT
# Best: 0.550000 using {'ff_dim': 4, 'lr': 1e-06} with secBERT
# Best: 0.512500 using {'ff_dim': 4, 'lr': 1e-05} with untrained BERT lol waht the fuck
param_grid = {
    'ff_dim': [4, 8, 16],
    'lr': [1e-6, 5e-6, 1e-5]
}

simple_grid = {}

grid = GridSearchCV(estimator=model, param_grid=simple_grid, cv=3)
grid_result = grid.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=1)

# Print the best parameters and corresponding accuracy
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# batch_size = 16
# epochs = 10

# history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
