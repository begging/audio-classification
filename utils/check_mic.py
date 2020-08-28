import pyaudio as pa
import numpy as np

p = pa.PyAudio()


rate_list = [8000, 11025, 16000, 22050, 32000, 37800, 44100, 47250, 48000,
             50000, 50400, 64000, 88200, 96000]

def print_device_info():
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            print("Input Device id ", i, " - ",
                p.get_device_info_by_host_api_device_index(0, i).get('name'),
                '(sample rates: {})'.format(str(get_supported_rate(i))))


def get_supported_rate(device_idx):
    available_rate_list = []
    for rate in rate_list:
        is_supported = False
        try:
            is_supported = p.is_format_supported(rate,
                                                 input_device=device_idx,
                                                 input_channels=1,
                                                 input_format=pa.paFloat32)

        except ValueError:
            pass

        else:
            if is_supported:
                available_rate_list.append(rate)

    return available_rate_list


def get_proper_rate(supported_rate_list, target_rate):

    supported_rate_arr = np.array(supported_rate_list)
    preprocessed = np.abs(np.subtract(supported_rate_arr, target_rate))
    min_idxs = np.where(preprocessed == np.amin(preprocessed))
    min_idx = min_idxs[0][-1]

    return supported_rate_list[min_idx]


if __name__ == '__main__':

    print_device_info()
