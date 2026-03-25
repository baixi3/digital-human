import numpy as np

import queue

from baseasr import BaseASR
from wav2lip import audio
# 每次收集一批新的音频帧，与保留下来的上下文帧拼成连续音频，提取 mel 频谱，
# 再按固定窗口切成多个 (80,16) 的 mel 特征块送给后续嘴型模型，同时只保留边界上下文以支持下一轮流式处理。
class LipASR(BaseASR):

    def run_step(self):
        for _ in range(self.batch_size*2):
            frame,type = self.get_audio_frame()
            self.frames.append(frame)
            self.output_queue.put((frame,type))
        # 上下文不够长不处理
        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames)
        mel = audio.melspectrogram(inputs)

        left = max(0, self.stride_left_size*80/50)
        right = min(len(mel[0]), len(mel[0]) - self.stride_right_size*80/50)
        mel_idx_multiplier = 80.*2/self.fps 
        mel_step_size = 16
        i = 0
        mel_chunks = []
        while i < (len(self.frames)-self.stride_left_size-self.stride_right_size)/2:
            start_idx = int(left + i * mel_idx_multiplier)
            #print(start_idx)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            else:
                mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1
        self.feat_queue.put(mel_chunks)
        
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
