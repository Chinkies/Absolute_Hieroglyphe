import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def test(seed, epochs, batch_size, col_index, col_name, image, folder):
    weight_file = f'models/model_s{seed}_b{batch_size}_e{epochs}_c{col_index}.weights.h5'

    # -------------------- Config --------------------

    img_height = 100
    img_width = 100

    tf.keras.utils.set_random_seed(seed)

    # -------------------- Datasets --------------------

    data_dir = pathlib.Path('dataset')

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = val_ds.class_names
    num_classes = len(class_names)

    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # -------------------- Layers --------------------

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

    # -------------------- Model load --------------------

    # seams useless, but actually it builds the model, where model.build() doesn't work
    model.predict(val_ds, batch_size=batch_size, verbose=0)

    model.load_weights(weight_file)

    # -------------------- Confusion Matrix --------------------

    y_pred = model.predict(val_ds, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    y_test_list = []
    for image_batch, labels_batch in val_ds:
        y_test_list.append(labels_batch)

    y_test = []
    for batch in y_test_list:
        for tensor in batch:
            y_test.append(int(tensor))

    conf_mat = confusion_matrix(y_test, y_pred)
    print("Confusion matrix :", conf_mat)

    fig = plt.figure()
    ax = fig.add_subplot()
    res = ax.imshow(np.array(conf_mat), cmap=plt.cm.Blues, interpolation='nearest')

    plt.xticks(range(num_classes), class_names)
    plt.yticks(range(num_classes), class_names)

    for x in range(len(conf_mat[0])):
        for tensor in range(len(conf_mat)):
            ax.annotate(str(conf_mat[x][tensor]), xy=(tensor, x),
                        horizontalalignment='center', verticalalignment='center')

    fig.colorbar(res)

    plt.title(f"Model confusion matrix for {col_name}")
    plt.savefig(f"plots/{col_name}/cmat_s{seed}_b{batch_size}_e{epochs}_c{col_index}.png")
    plt.close()

    # -------------------- Tests --------------------

    prediction = test_image(model, batch_size, (img_width, img_height), image)
    print(f"predicted class for {col_name} of {image} is", class_names[prediction])

    predictions = test_folder(model, batch_size, (img_width, img_height), folder)
    images = os.listdir(folder)
    for i in range(len(predictions)):
        print(f"predicted class for {col_name} of file {images[i]} is", class_names[predictions[i]])


def test_image(model, batch_size, img_size, file):
    image = tf.keras.utils.load_img(file, target_size=img_size)
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])

    y_pred = model.predict(input_arr, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred[0]


def test_folder(model, batch_size, img_size, folder):
    image_arr = []
    for file in os.listdir(folder):
        image = tf.keras.utils.load_img(folder + '/' + file, target_size=img_size)
        input_arr = tf.keras.utils.img_to_array(image)
        image_arr.append(input_arr)

    y_pred = model.predict(np.array(image_arr), batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred
