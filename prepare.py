import os

import dm_config as cfg
from api.convertors.mp3_to_wav import convert_from_mp3_to_wav_and_save_to_path
from api.convertors.wav_to_spectrogram import convert_wav_to_spectrogram_and_save_to_path


def main():
    dir_arr = os.listdir(path=cfg.DIR_PATH_TO_IN_MP3)
    for dir in dir_arr:
        convert_from_mp3_to_wav_and_save_to_path(from_path=cfg.DIR_PATH_TO_IN_MP3 + dir + "\\",
                                                 to_path=cfg.DIR_PATH_TO_OUT_WAV + dir + "\\")
        convert_wav_to_spectrogram_and_save_to_path(from_path=cfg.DIR_PATH_TO_OUT_WAV + dir + "\\",
                                                    to_path=cfg.DIR_PATH_TO_OUT_SPECTROGRAM + dir + "\\")


main()
