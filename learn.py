import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from tensorflow.python.keras import backend
from tensorflow.python.keras.callbacks import ModelCheckpoint

from api.model.model_cnn import build_model_and_get
from dm_config import DIR_PATH_TO_OUT_MODAL_WEIGHT, DIR_PATH_TO_OUT_SPECTROGRAM, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT

y_targets = {
    "schubert": [0, 1],
    "mozart": [1, 0],
    "chopin": [0, 1]
}


def main():
    main_dir_arr = os.listdir(DIR_PATH_TO_OUT_SPECTROGRAM)
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for m_dir in main_dir_arr:
        dir_arr = os.listdir(DIR_PATH_TO_OUT_SPECTROGRAM + m_dir)
        i = 0
        if 'mozart' in m_dir or 'chopin' in m_dir:
            for c_dir in dir_arr:
                x = image.imread(DIR_PATH_TO_OUT_SPECTROGRAM + m_dir + "\\" + c_dir)
                x = x[:, :, :-1]
                y = y_targets.get(m_dir)
                x_train.append(x)
                y_train.append(y)
                i = i + 1
                if i < 10000:
                    x_test.append(x)
                    y_test.append(y)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    if backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)
        x_test = x_test.reshape(x_test.shape[0], 3, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)
        input_shape = (3, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)
    else:
        x_train = x_train.reshape(x_train.shape[0], SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, 3)
        x_test = x_test.reshape(x_test.shape[0], SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, 3)
        input_shape = (SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    model = build_model_and_get(input_shape)
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=40,
        batch_size=128,
        callbacks=[
            ModelCheckpoint(filepath=DIR_PATH_TO_OUT_MODAL_WEIGHT + 'model_{epoch:00003d}',
                            save_best_only=True)
        ]
    )
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


main()
