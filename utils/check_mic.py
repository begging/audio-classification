import pyaudio as pa

p = pa.PyAudio()


rate_list = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000]

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


if __name__ == '__main__':

    print_device_info()
