import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# import tensorflow_datasets as tfds
import pathlib

seed = 1
tf.keras.utils.set_random_seed(seed)
data_dir = pathlib.Path('dataset')

image_count_jpg = len(list(data_dir.glob('*/*.jpg')))
image_count_png = len(list(data_dir.glob('*/*.png')))
print(image_count_jpg + image_count_png)

# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[0]))

# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[1]))

batch_size = 16
epochs = 8
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

print(len(val_ds))
for i in val_ds:
    print(i)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1): # take first batch
    print(len(images))
    print(labels)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

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

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Test du modèle sur le jeu d'évaluation
score = model.evaluate(val_ds, verbose=0)

print(f"Test loss : {score[0]:4.4f}")
print(f"Test accuracy : {score[1]:4.4f}")

# Historique de la précision du modèle sur les jeux d'entraînement et de validation
print(model.metrics_names)

if not os.path.exists("plots"):
    os.mkdir("plots")

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
plt.plot(history.history['loss'], scaley=False, color="tab:green")
plt.plot(history.history['val_loss'], scaley=False, color="tab:red")
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f"plots/loss_s{seed}_b{batch_size}_e{epochs}_c9.png")
plt.close()

exit(0)

# Matrice de confusion
class Estimator:
    _estimator_type = ''
    _classes = []
    def __init__(self, _model, classes):
        self.model = _model
        self._estimator_type = 'classifier'
        self._classes = classes
    def predict(self, x):
        y_prob = self.model.predict(x)
        y_pred = y_prob.argmax(axis = 1)
        return y_pred

# class_names_nb = [0, 1, 2, 3, 4]

classifier = Estimator(model, class_names)

x_test = list(map(lambda x: x[0], val_ds))
y_test = list(map(lambda x: x[1], val_ds))

ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)
plt.show()
