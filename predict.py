import os

import numpy as np
from matplotlib import image
from tensorflow.python.keras import backend

from api.dir_op import delete_all_files_in_dir, delete_all_files_in_dirs
from api.convertors.mp3_to_wav import convert_from_mp3_to_wav_and_save_to_path
from api.convertors.wav_to_spectrogram import convert_wav_to_spectrogram_and_save_to_path, \
    convert_wav_to_sftf_spectrogram_and_save_to_path
from api.model.model_cnn import build_model_and_get
from dm_config import DIR_PATH_TO_OUT_MODAL_WEIGHT, DIR_PATH_TO_OUT_SPECTROGRAM_PREDICT, DIR_PATH_TO_IN_MP3_PREDICT, \
    DIR_PATH_TO_OUT_WAV_PREDICT, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT


def main():
    delete_all_files_in_dirs(paths=[DIR_PATH_TO_OUT_SPECTROGRAM_PREDICT, DIR_PATH_TO_OUT_WAV_PREDICT])

    convert_from_mp3_to_wav_and_save_to_path(from_path=DIR_PATH_TO_IN_MP3_PREDICT,
                                             to_path=DIR_PATH_TO_OUT_WAV_PREDICT)
    convert_wav_to_spectrogram_and_save_to_path(from_path=DIR_PATH_TO_OUT_WAV_PREDICT,
                                                to_path=DIR_PATH_TO_OUT_SPECTROGRAM_PREDICT)
    spectrogram_files_array = os.listdir(DIR_PATH_TO_OUT_SPECTROGRAM_PREDICT)
    x_train = []
    for spectrogram_file_name in spectrogram_files_array:
        data = image.imread(DIR_PATH_TO_OUT_SPECTROGRAM_PREDICT + spectrogram_file_name)
        x_train.append(data[:, :, :-1])

    x_train = np.array(x_train)
    if backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)
        input_shape = (3, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)
    else:
        x_train = x_train.reshape(x_train.shape[0], SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, 3)
        input_shape = (SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT, 3)

    x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_test /= 255
    model = build_model_and_get(input_shape)
    model.load_weights(DIR_PATH_TO_OUT_MODAL_WEIGHT + 'model_004')
    predict = model.predict(x_train)
    i_count = 0
    i_sum = 0

    for xarr in predict:
        i_count = i_count + 1
        i_sum = i_sum + xarr[0]
    print('{:0.5f}'.format(i_sum / i_count))


main()
