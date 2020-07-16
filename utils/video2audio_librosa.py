import numpy as np
import librosa
import librosa.display
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure



def melspectrogram(audio, sr=44100, n_mels=128):
    return librosa.amplitude_to_db(librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels))

def show_melspectrogram(mel, sr=44100):
    plt.figure(figsize=(14,4))
    librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Log mel spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()

    plt.show()


# class ffmpegProcessor:
#     def __init__(self):
#         self.cmd = './ffmpeg'

#     def extract_audio(self, filename):
#         try:
#             out, err = (
#                 ffmpeg
#                 .input(filename)
#                 .output('-', format='f32le', acodec='pcm_f32le', ac=1, ar='16000')
#                 .run(cmd=self.cmd, capture_stdout=True, capture_stderr=True)
#             )
#         except Error as err:
#             print(err.stderr)
#             raise

#         return np.frombuffer(out, np.float32)


# # ap = ffmpegProcessor()
# # audio = ap.extract_audio('_tmp/sohyang.mp4')

sample_rate = 16000
(waveform, _) = librosa.core.load('_tmp/sample.mkv', sr=sample_rate, mono=True)

librosa.output.write_wav('_tmp/sample.wav', waveform, sample_rate)

show_melspectrogram(melspectrogram(waveform))
