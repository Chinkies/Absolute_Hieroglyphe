import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


# -------------------- Config --------------------

seed = 92
batch_size = 16
epochs = 32
img_height = 180
img_width = 180

tf.keras.utils.set_random_seed(seed)

# -------------------- Datasets --------------------

data_dir = pathlib.Path('dataset')

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

# -------------------- Normalization --------------------

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------------------- Layers --------------------

# TODO jouer avec les layers -> GlyphNet
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

# -------------------- Model training --------------------

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    shuffle=False
)



# -------------------- Plots --------------------

if not os.path.exists("plots"):
    os.mkdir("plots")

score = model.evaluate(val_ds, verbose=0)

# Accuracy
plt.plot(history.history['accuracy'], scaley=False, color="tab:blue")
plt.plot(history.history['val_accuracy'], scaley=False, color="tab:orange")
plt.title("Model accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['train', 'validation'], loc='upper left')
# TODO indiquer la colonne choisie
plt.savefig(f"plots/accu_s{seed}_b{batch_size}_e{epochs}_c9.png")
plt.close()

# Loss
plt.plot(history.history['loss'], color="tab:green")
plt.plot(history.history['val_loss'], color="tab:red")
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f"plots/loss_s{seed}_b{batch_size}_e{epochs}_c9.png")
plt.close()

# -------------------- Confusion Matrix --------------------

y_test_list = []
for image_batch, labels_batch in val_ds:
    y_test_list.append(labels_batch)

y_test = []
for batch in y_test_list:
    for tensor in batch:
        y_test.append(int(tensor))

y_pred = model.predict(val_ds, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1)

conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion matrix :", conf_mat)

fig = plt.figure()
ax = fig.add_subplot()
res = ax.imshow(np.array(conf_mat), cmap=plt.cm.jet, interpolation='nearest')

alphabet = [i for i in range(num_classes)]
plt.xticks(range(num_classes), alphabet[:num_classes])
plt.yticks(range(num_classes), alphabet[:num_classes])

for x in range(num_classes):
    for tensor in range(num_classes):
        ax.annotate(str(conf_mat[x][tensor]), xy=(tensor, x),
                    horizontalalignment='center', verticalalignment='center')

fig.colorbar(res)

plt.savefig(f"plots/cmat_s{seed}_b{batch_size}_e{epochs}_c9.png")
