import os
import sys

import argparse

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio as pa
import threading
import time
import torch

import config
from models import *
from utilities import create_folder, get_filename, show_text_on_image
from pytorch_utils import move_data_to_device


def audio_tagging(conf):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    window_size = conf['window_size']
    hop_size = conf['hop_size']
    mel_bins = conf['mel_bins']
    fmin = conf['fmin']
    fmax = conf['fmax']
    model_type = conf['model_type']
    checkpoint_path = conf['checkpoint_path']
    audio_path = conf['audio_path']

    if conf['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sample_rate = config.sample_rate
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
        classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    print(waveform.shape)
    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels


def sound_event_detection(conf):
    """Inference sound event detection result of an audio clip.
    """

    # Arugments & parameters
    window_size = conf['window_size']
    hop_size = conf['hop_size']
    mel_bins = conf['mel_bins']
    fmin = conf['fmin']
    fmax = conf['fmax']
    model_type = conf['model_type']
    checkpoint_path = conf['checkpoint_path']
    audio_path = conf['audio_path']
    tok_k = conf['top_k']

    if conf['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sample_rate = config.sample_rate
    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Paths
    fig_path = os.path.join(
        'results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
        classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    # waveform = waveform[None, :] # (1, audio_length)
    waveform = waveform[None, :int(len(waveform)/7*2)] # (1, audio_length)

    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]

    print(framewise_output.shape)
    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]

    if conf['save_plot']:
        sorted_indexes = np.argsort(clipwise_output)[::-1]

        top_k = 10  # Show top results
        top_result_mat = clipwise_output[sorted_indexes[:top_k]]
        print(top_result_mat)
        print(sorted_indexes[:top_k])

        print('Sound event detection result (time_steps x classes_num): {}'.format(
            framewise_output.shape))

        sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

        top_k = 10  # Show top results
        top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]
        """(time_steps, top_k)"""

        # Plot result
        stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size,
            hop_length=hop_size, window='hann', center=True)
        frames_num = stft.shape[-1]

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
        axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
        axs[0].set_ylabel('Frequency bins')
        axs[0].set_title('Log spectrogram')
        axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
        axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
        axs[1].yaxis.set_ticks(np.arange(0, top_k))
        axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
        axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        axs[1].set_xlabel('Seconds')
        axs[1].xaxis.set_ticks_position('bottom')

        plt.tight_layout()
        plt.savefig(fig_path)
        print('Save sound event detection visualization to {}'.format(fig_path))

    return framewise_output, clipwise_output, labels


def sound_event_detection_with_mic_input(conf):

    chunk = conf['chunk']
    device_index = conf['device_index']
    channels = conf['channels']
    rate = conf['rate']
    record_seconds = conf['record_seconds']
    step_seconds = conf['step_seconds']
    dtype = pa.paFloat32

    mutex = threading.Lock()
    waveform = None
    thread_stop = False

    def mic_thread_func():
        nonlocal waveform
        nonlocal thread_stop

        p = pa.PyAudio()

        stream = p.open(input_device_index=device_index,
                        format=dtype,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

        len_to_record = int(round(record_seconds*rate/chunk))
        len_to_step = int(round(step_seconds*rate/chunk))

        frames = []

        print('Recording ...')
        while not thread_stop:
            data = stream.read(chunk)
            frames.append(np.frombuffer(data, dtype=np.float32))

            if len(frames) >= len_to_step:
                if len(frames) >= len_to_record:
                    with mutex:
                        waveform = np.hstack(frames[-len_to_record:])
                        waveform = waveform[None, :]

                    if len_to_step >= len_to_record:
                        frames = []
                    else:
                        frames = frames[len_to_step:]

        stream.stop_stream()
        stream.close()
        p.terminate()

    # Arugments & parameters
    window_size = conf['window_size']
    hop_size = conf['hop_size']
    mel_bins = conf['mel_bins']
    fmin = conf['fmin']
    fmax = conf['fmax']
    model_type = conf['model_type']
    checkpoint_path = conf['checkpoint_path']
    top_k = conf['top_k']
    image_size = conf['image_size']

    if conf['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU mode: True')
    else:
        device = torch.device('cpu')
        print('GPU mode: False')

    sample_rate = config.sample_rate
    classes_num = config.classes_num
    labels = config.labels

    print('Model loading ({}) ... '.format(model_type))
    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
        classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel

    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    mic_thread = threading.Thread(target=mic_thread_func)
    mic_thread.start()

    while True:
        # Load audio
        if waveform is None:
            time.sleep(0.01)
            continue

        device_waveform = move_data_to_device(waveform, device)
        with mutex:
            waveform = None

        # Forward
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(device_waveform, None)

        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
        """(classes_num,)"""

        sorted_indexes = np.argsort(clipwise_output)[::-1]

        # Print audio tagging top probabilities
        result_text_list = []
        for k in range(top_k):

            text = '{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
                    clipwise_output[sorted_indexes[k]])

            result_text_list.append(text)

            # print(text)

        result_window_name = "Sound Detection Result"
        ret = show_text_on_image(result_text_list, result_window_name,
            image_size)

        # if ret in [26, ord('q'), ord('Q')]:
        #     thread_stop = True
        #     cv2.destroyWindow(result_window_name)
        #     break

    if mic_thread.is_alive():
        mic_thread.join()

    return clipwise_output, labels


if __name__ == '__main__':
    pass
