import os
import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf


def train(seed, epochs, batch_size, col_index, col_name):
    # -------------------- Config --------------------

    img_height = 100
    img_width = 100
    weight_file = f'models/model_s{seed}_b{batch_size}_e{epochs}_c{col_index}.weights.h5'

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

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    # -------------------- Layers --------------------

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        # tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # tf.keras.layers.ReLU(),
        # tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # tf.keras.layers.ReLU(),
        #
        # tf.keras.layers.SeparableConv2D(128, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ReLU(),
        # tf.keras.layers.SeparableConv2D(128, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # tf.keras.layers.ReLU(),
        #
        # tf.keras.layers.SeparableConv2D(128, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ReLU(),
        # tf.keras.layers.SeparableConv2D(128, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # tf.keras.layers.ReLU(),

        # TODO finir architecture GlyphNet
        # tf.keras.layers.SeparableConv2D(256, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ReLU(),
        # tf.keras.layers.SeparableConv2D(256, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # tf.keras.layers.ReLU(),
        #
        # tf.keras.layers.SeparableConv2D(256, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ReLU(),
        # tf.keras.layers.SeparableConv2D(256, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # tf.keras.layers.ReLU(),

        # tf.keras.layers.SeparableConv2D(512, kernel_size=3, strides=(1, 1)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.ReLU(),
        # tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Dropout(0.15),
        # tf.keras.layers.Softmax(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # -------------------- Model training --------------------

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        shuffle=False
    )

    if not os.path.isdir("models"):
        os.mkdir("models")

    model.save_weights(weight_file)

    # -------------------- Plots --------------------

    if not os.path.isdir(f"plots/{col_name}"):
        os.mkdir(f"plots/{col_name}")

    # Accuracy
    plt.plot(history.history['accuracy'], scaley=False, color="tab:blue")
    plt.plot(history.history['val_accuracy'], scaley=False, color="tab:orange")
    plt.title(f"Model accuracy for {col_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"plots/{col_name}/accu_s{seed}_b{batch_size}_e{epochs}_c{col_index}.png")
    plt.close()

    # Loss
    plt.plot(history.history['loss'], color="tab:green")
    plt.plot(history.history['val_loss'], color="tab:red")
    plt.title(f"Model loss for {col_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"plots/{col_name}/loss_s{seed}_b{batch_size}_e{epochs}_c{col_index}.png")
    plt.close()
