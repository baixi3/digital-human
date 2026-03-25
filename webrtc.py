import asyncio
import logging
import threading
import time
from typing import Optional, Set, Tuple, Union

import fractions
from av.frame import Frame
from av.packet import Packet
from aiortc import MediaStreamTrack

AUDIO_PTIME = 0.020  # 每个音频包时长 20ms
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 25  # 25fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

logging.basicConfig()
logger = logging.getLogger(__name__)


class PlayerStreamTrack(MediaStreamTrack):
    """
    这个媒体轨背后不是自己直接产生数据，是由一个渲染工作线程在后台产数据。
    """
    _start: float
    _timestamp: int

    def __init__(self, player, kind):
        super().__init__()
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue()
        if self.kind == "video":
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == "video":
            if hasattr(self, "_timestamp"):
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                print("video start:", self._start)
            return self._timestamp, VIDEO_TIME_BASE

        if hasattr(self, "_timestamp"):
            self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
            wait = self._start + (self._timestamp / SAMPLE_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
            print("audio start:", self._start)
        return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        self._player._start(self)
        frame = await self._queue.get()
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == "video":
            self.totaltime += time.perf_counter() - self.lasttime
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                print(f"------平均帧率 : {self.framecount / self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime = 0
        return frame

    def stop(self):
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None


def player_worker_thread(quit_event, loop, renderer, audio_track, video_track):
    renderer.render(quit_event, loop, audio_track, video_track)


class HumanPlayer:
    def __init__(self, real):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")
        self.__container = real

    @property
    def audio(self) -> MediaStreamTrack:
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("开始工作线程")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    asyncio.get_event_loop(),
                    self.__container,
                    self.__audio,
                    self.__video,
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"HumanPlayer {msg}", *args)
