import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))

import argparse
import logging
from datetime import datetime
from pprint import pprint

import cv2
import numpy as np
import pyaudio as pa

from pytorch.inference import SoundDetector
from utils.utilities import parse_config, create_logger


if __name__ == '__main__':
    conf = parse_config()


    # set logger
    log = create_logger()

    if 'file' in conf['output_mode']:
        if conf['output_file_path']:
            file_name = conf['output_file_path']
        else:
            file_name = 'result.log'
        log.addHandler(logging.FileHandler(file_name))

    sh = logging.StreamHandler()

    if 'stdout' not in conf['output_mode']:
        sh.setLevel(logging.INFO)

    log.addHandler(sh)
    log.info('<{}>'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    sd = SoundDetector(conf, log)
    sd.run()
