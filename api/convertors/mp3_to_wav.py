import os

from pydub import AudioSegment

from dm_config import DIR_PATH_TO_OUT_WAV, DIR_PATH_TO_IN_MP3


def convert_from_mp3_to_wav_and_save_to_path(from_path=DIR_PATH_TO_IN_MP3, to_path=DIR_PATH_TO_OUT_WAV):
    if not from_path: return -1
    mp3_files_array = os.listdir(from_path)
    for mp3_file_name in mp3_files_array:
        mp3_file = AudioSegment.from_mp3(from_path + mp3_file_name)
        len_mp3_file = len(mp3_file)
        number_of_parts = int(len_mp3_file / 1000)
        for i in range(number_of_parts):
            start_split = i * 1000
            end_split = (i + 1) * 1000
            split_mp3_file = mp3_file[start_split:end_split]
            wav_file_name = mp3_file_name.replace('.mp3', str(i) + '.wav')
            split_mp3_file.export(to_path + wav_file_name, format='wav')
