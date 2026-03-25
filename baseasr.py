import multiprocessing as mp
import queue
from queue import Queue

import numpy as np


class BaseASR:
    def __init__(self, opt):
        self.fps = opt.fps
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps
        self.queue = Queue()
        self.output_queue = mp.Queue()
        self.batch_size = opt.batch_size
        self.frames = []
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        self.feat_queue = mp.Queue(2)

    def pause_talk(self):
        self.queue.queue.clear()

    def put_audio_frame(self, audio_chunk):
        self.queue.put(audio_chunk)

    def get_audio_frame(self):
        try:
            frame = self.queue.get(block=True, timeout=0.01)
            return frame, 0
        except queue.Empty:
            return np.zeros(self.chunk, dtype=np.float32), 1

    def warm_up(self):
        warm_up_size = self.stride_left_size + self.stride_right_size
        for _ in range(warm_up_size):
            audio_frame, frame_type = self.get_audio_frame()
            self.frames.append(audio_frame)
            self.output_queue.put((audio_frame, frame_type))

        for _ in range(self.stride_left_size):
            self.output_queue.get()

    def run_step(self):
        raise NotImplementedError
