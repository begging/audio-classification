import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))

import argparse
from pprint import pprint

import cv2
import numpy as np
import pyaudio as pa
import wave

from utils.utilities import parse_config
from pytorch.inference import sound_event_detection
from pytorch.inference import audio_tagging
from pytorch.inference import sound_event_detection_with_mic_input



def sound_classification_on_audio(conf):
    (clipwise_output, labels) = audio_tagging(conf)

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
            clipwise_output[sorted_indexes[k]]))

    # Do something


def sound_detection_on_audio(conf):
    (framewise_output, clipwise_output, labels) = sound_event_detection(conf)

    # Do something


def sound_detection_on_video(conf):
    # Sound event detection
    (framewise_output, clipwise_output, labels) = sound_event_detection(conf)

    # Add detected results text to video
    add_text_to_video_and_save(
        framewise_output=framewise_output,
        labels=labels,
        conf=conf)


def add_text_to_video_and_save(framewise_output, labels, conf):

    topk = conf['top_k']    # Number of sound classes to show
    out_video_path = conf['out_video_path']
    video_path = conf['video_path']
    ffmpeg_path = conf['ffmpeg_path']

    sed_frames_per_second = 100

    # Paths
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

    tmp_video_path = '_tmp/tmp.avi'
    os.makedirs('_tmp', exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('frame_width: {}, frame_height: {}, fps: {}'.format(
        frame_width, frame_height, fps))

    sed_frames_per_video_frame = sed_frames_per_second / float(fps)

    out = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*'H264'),
        fps, (frame_width, frame_height))

    # For each frame select the top classes
    sorted_indexes = np.argsort(framewise_output, axis=-1)[:, -1 : -topk - 1 : -1]

    """(frames_num, topk)"""
    rect_coord = (int(0.4*frame_width), int(frame_height/20*(topk+1)))

    n = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # End of video
        if frame is None:
            break

        cv2.rectangle(frame, (0, 0), rect_coord, (0, 0, 0), -1)

        m = int(n * sed_frames_per_video_frame)
        for k in range(0, topk):

            if m >= len(framewise_output):
                break

            text = '{}: {:.3f}'.format(
                cut_words(labels[sorted_indexes[m, k]]),
                framewise_output[m, sorted_indexes[m, k]])

            # Add text to frames
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = frame_width / 1280
            bottomLeftCornerOfText = (5, int(0.05*frame_height + k*(frame_height/20)))
            fontColor = (0,255,0)
            lineType = 2

            cv2.putText(
                frame,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

        # Write frame to video
        out.write(frame)

        n += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    os.system('{} -loglevel panic -y -i {} "{}"'.format(
        ffmpeg_path,
        tmp_video_path,
        out_video_path))

    print('Write silent video with text to {}'.format(out_video_path))


def cut_words(lb, max_len=20):
    """Cut label to max_len.
    """
    words = lb.split(', ')
    new_lb = ''
    for word in words:
        if len(new_lb + ', ' + word) > max_len:
            break
        new_lb += ', ' + word
    new_lb = new_lb[2 :]

    if len(new_lb) == 0:
        new_lb = words[0]

    return new_lb


def sound_event_detection_real_time(conf):
    (clipwise_output, labels) = sound_event_detection_with_mic_input(conf)


if __name__ == '__main__':
    conf = parse_config()

    # filewise audio classificaiton on an audio file
    if conf['mode'] == 0:
        sound_classification_on_audio(conf)

    # framewise audio detection on an audio file
    elif conf['mode'] == 1:
        sound_detection_on_audio(conf)

    # framewise audio detection on a video file
    elif conf['mode'] == 2:
        sound_detection_on_video(conf)

    elif conf['mode'] == 3:
        sound_event_detection_real_time(conf)
