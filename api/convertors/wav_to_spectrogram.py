import os

import librosa as lsa
import matplotlib.pyplot as plt
import numpy as np
from librosa import display as lsa_dly

from dm_config import DIR_PATH_TO_OUT_SPECTROGRAM, DIR_PATH_TO_OUT_WAV, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT


def convert_wav_to_spectrogram_and_save_to_path(from_path=DIR_PATH_TO_OUT_WAV, to_path=DIR_PATH_TO_OUT_SPECTROGRAM):
    wav_files_array = os.listdir(from_path)
    # spectrogram_array = []
    # i = 0
    for wav_file_name in wav_files_array:
        y, sr = lsa.load(from_path + wav_file_name)
        mel_spectrogram = lsa.feature.melspectrogram(y=y)
        mel_spectrogram_to_db = lsa.power_to_db(mel_spectrogram, ref=np.max)
        fig = plt.figure(figsize=(SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT), dpi=1, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        lsa_dly.specshow(mel_spectrogram_to_db)

        # for example save
        # names = tempfile._get_candidate_names()
        # name = next(names)
        out_spectrogram_name = wav_file_name.replace('.wav', '.png')
        fig.savefig(to_path + out_spectrogram_name, bbox_inches='tight', pad_inches=0)
        # end fig.canvas.draw() canvas = fig.canvas.tostring_rgb() np_array_with_filters = np.fromstring(canvas,
        # dtype='uint8').reshape((3, SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT)) spectrogram_array.append(
        # np_array_with_filters) i = i + 1
        plt.close(fig)

    # return np.array(spectrogram_array)


def convert_wav_to_sftf_spectrogram_and_save_to_path(from_path=DIR_PATH_TO_OUT_WAV,
                                                     to_path=DIR_PATH_TO_OUT_SPECTROGRAM):
    wav_files_array = os.listdir(from_path)

    for wav_file_name in wav_files_array:
        y, sr = lsa.load(from_path + wav_file_name)
        chroma_stft = lsa.feature.chroma_stft(y, sr=sr)
        fig = plt.figure(figsize=(SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT), dpi=1, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        lsa_dly.specshow(chroma_stft, cmap='coolwarm')
        out_spectrogram_name = wav_file_name.replace('.wav', '.png')
        fig.savefig(to_path + out_spectrogram_name, bbox_inches='tight', pad_inches=0)

        plt.close(fig)
