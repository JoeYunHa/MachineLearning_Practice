import tensorflow as tf

shakespear_url = "https://homl.info/shakespeare"
filepath = tf.keras.utils.get_file("shakespeare.txt",shakespear_url)
with open(filepath) as f:
    shakespear_text = f.read()

text_vec_layer = tf.keras.layers.TextVectorization(split='character',
                                                   standardize='lower')
text_vec_layer.adapt([shakespear_text])
encoded = text_vec_layer([shakespear_text])[0]

encoded-=2 # token 0 -> padding token 1 -> unknown char => not use
n_tokens = text_vec_layer.vocabulary_size() - 2
dataset_size = len(encoded)
# print(dataset_size) => 1,115,394

def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)

length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1_000_000], length=length, shuffle=True,
                       seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000], length=length)
test_set = to_dataset(encoded[1_060_000:], length=length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation='softmax')
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy", save_best_only=True
    )
history = model.fit(train_set, validation_data=valid_set, epochs=10,
                    callbacks=[model_ckpt])

shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),
    model
    ])

y_poba = shakespeare_model.predict(["To be or not to "])[0,-2]
y_pred = tf.argmax(y_poba)
text_vec_layer.get_vocabulary()[y_pred + 2]

