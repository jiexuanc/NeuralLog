from data_loader import load_darpa
from neurallog.models.transformers import transformer_classifer

embed_dim = 768  # Embedding size for each token
num_heads = 12  # Number of attention heads
ff_dim = 2048  # Hidden layer size in feed forward network inside transformer
max_len = 20

(x_train, y_train), (x_test, y_test) = load_darpa(npz_file="data-bert.npz")
model = transformer_classifer(embed_dim, ff_dim, max_len, num_heads)