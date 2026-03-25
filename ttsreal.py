import asyncio
import queue
import time
from enum import Enum
from io import BytesIO
from threading import Thread

import edge_tts
import numpy as np
import resampy
import soundfile as sf


class State(Enum):
    RUNNING = 0
    PAUSE = 1


class BaseTTS:
    def __init__(self, opt, parent):
        self.opt = opt
        self.parent = parent
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps
        self.msgqueue = queue.Queue()
        self.state = State.RUNNING

    def pause_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg):
        self.msgqueue.put(msg)

    def render(self, quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()

    def process_tts(self, quit_event):
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            self.state = State.RUNNING
            self.txt_to_audio(msg)

        print("ttsreal停止")

    def emit_audio_stream(self, audio_stream, source_rate):
        for chunk in audio_stream:
            if not chunk or self.state != State.RUNNING:
                continue

            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
            pcm = resampy.resample(x=pcm, sr_orig=source_rate, sr_new=self.sample_rate)
            self.emit_float_samples(pcm)

    def emit_float_samples(self, stream):
        stream_length = stream.shape[0]
        index = 0
        while stream_length >= self.chunk and self.state == State.RUNNING:
            self.parent.put_audio_frame(stream[index : index + self.chunk])
            stream_length -= self.chunk
            index += self.chunk

    def txt_to_audio(self, msg):
        raise NotImplementedError


class EdgeTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.voice_name = "zh-CN-YunxiaNeural" # 设置要用的语音角色。
        self.input_stream = BytesIO()

    def txt_to_audio(self, msg):
        start = time.time()
        self._run_edge_stream(msg)
        print(f"-------edge tts 耗时:{time.time() - start:.4f}s")

        self.input_stream.seek(0)
        audio_stream = self._read_wav_stream(self.input_stream)
        self.emit_float_samples(audio_stream)
        self.input_stream.seek(0)
        self.input_stream.truncate()

    def _read_wav_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)
        print(f"[INFO]tts audio stream {sample_rate}: {stream.shape}")
        stream = stream.astype(np.float32)

        if stream.ndim > 1: # 立体声
            print(f"[WARN] {stream.shape[1]} 只取地一声道.")
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            print(f"[WARN] 声音采样率 {sample_rate}, 重设为 {self.sample_rate}.")
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def _run_edge_stream(self, text):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._stream_edge_tts(text))
        finally:
            loop.close()

    async def _stream_edge_tts(self, text: str):
        communicate = edge_tts.Communicate(text, self.voice_name)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio" and self.state == State.RUNNING:
                self.input_stream.write(chunk["data"])
