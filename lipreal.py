import asyncio
import glob
import multiprocessing as mp
import os
import pickle
import queue
import time
from threading import Thread

import cv2
import numpy as np
import torch
from av import AudioFrame, VideoFrame
from tqdm import tqdm

from basereal import BaseReal
from lipasr import LipASR
from wav2lip.models import Wav2Lip

MODEL_PATH = "./models/wav2lip.pth"
IMAGE_GLOB = "*.[jpJP][pnPN]*[gG]"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"用 {DEVICE} 推理.")


def _load_checkpoint(checkpoint_path):
    if DEVICE == "cuda":
        return torch.load(checkpoint_path)
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

# 加载模型
def load_model(path):
    model = Wav2Lip()
    print(f"path: {path}")
    checkpoint = _load_checkpoint(path)
    state_dict = checkpoint["state_dict"]
    clean_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model = model.to(DEVICE)
    return model.eval()


def list_image_files(image_dir):
    # 返回路径列表
    image_paths = glob.glob(os.path.join(image_dir, IMAGE_GLOB))
    return sorted(image_paths, key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))


def read_images(image_paths):
    frames = []
    print("读图像...")
    for image_path in tqdm(image_paths):
        frames.append(cv2.imread(image_path))
    return frames

# 数字人来回跑不会闪现一下很尴尬
def mirror_index(size, index):
    turn = index // size
    offset = index % size
    if turn % 2 == 0:
        return offset
    return size - offset - 1


def build_model_inputs(face_frames, mel_batch):
    image_batch = np.asarray(face_frames)
    mel_batch = np.asarray(mel_batch)

    masked_batch = image_batch.copy()
    masked_batch[:, image_batch.shape[1] // 2 :] = 0 # 将图像的下半脸遮住(B,H,W,C)

    image_batch = np.concatenate((masked_batch, image_batch), axis=3) / 255.0
    # 方便和卷积网络输入规范一致
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
    # 处理成模型输入
    image_batch = torch.FloatTensor(np.transpose(image_batch, (0, 3, 1, 2))).to(DEVICE)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(DEVICE)
    return image_batch, mel_batch


def read_audio_batch(audio_out_queue, batch_size):
    audio_frames = []
    has_speech = False
    for _ in range(batch_size * 2):
        frame, frame_type = audio_out_queue.get()
        audio_frames.append((frame, frame_type))
        if frame_type == 0:
            has_speech = True   
    return audio_frames, has_speech

# 跑嘴型推理的子进程
# 在渲染开关允许的情况下，持续从音频队列中取一批音频特征和音频帧；如果检测到人声，就结合对应人脸帧调用 Wav2Lip 模型生成嘴型结果；如果没有人声，就直接输出空结果；最后把结果帧、对应底图索引和音频片段一起送入结果队列。
def inference_worker(render_event, batch_size, face_images_path, audio_feat_queue, audio_out_queue, result_queue):
    model = load_model(MODEL_PATH)
    face_frames = read_images(list_image_files(face_images_path))
    frame_count = len(face_frames)
    frame_index = 0 # 当前播放到第几帧
    infer_frames = 0
    infer_time = 0.0

    print("开始推理")
    while True:
        if not render_event.is_set():
            time.sleep(1)
            continue
        try:
            mel_batch = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue

        audio_frames, has_speech = read_audio_batch(audio_out_queue, batch_size)
        if not has_speech:
            for batch_index in range(batch_size):
                result_queue.put(
                    (
                        None,
                        mirror_index(frame_count, frame_index),
                        audio_frames[batch_index * 2 : batch_index * 2 + 2],
                    )
                )
                frame_index += 1
            continue

        batch_face_frames = []
        for batch_index in range(batch_size):
            current_index = mirror_index(frame_count, frame_index + batch_index)
            batch_face_frames.append(face_frames[current_index])

        start = time.perf_counter()
        image_batch, mel_batch = build_model_inputs(batch_face_frames, mel_batch)
        with torch.no_grad():
            predictions = model(mel_batch, image_batch)
        predictions = predictions.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        infer_time += time.perf_counter() - start
        infer_frames += batch_size
        if infer_frames >= 100:
            print(f"------ 平均推理速度 : {infer_frames / infer_time:.4f}")
            infer_frames = 0
            infer_time = 0.0

        for batch_index, result_frame in enumerate(predictions):
            result_queue.put(
                (
                    result_frame,  # 嘴部推理结果
                    mirror_index(frame_count, frame_index), # 对应贴回去完整素材帧索引
                    audio_frames[batch_index * 2 : batch_index * 2 + 2], # 与之对应的两个音频小帧
                )
            )
            frame_index += 1


class LipReal(BaseReal):
    def __init__(self, opt):
        super().__init__(opt)
        self.avatar_id = opt.avatar_id
        self.batch_size = opt.batch_size
        self.avatar_path = f"./data/avators/{self.avatar_id}"
        self.full_images_path = f"{self.avatar_path}/full_imgs"
        self.face_images_path = f"{self.avatar_path}/face_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.result_queue = mp.Queue(self.batch_size * 2)
        self.render_event = mp.Event()

        self._load_avatar()

        self.asr = LipASR(opt)
        self.asr.warm_up()

        self.inference_process = mp.Process(
            target=inference_worker,
            args=(
                self.render_event,
                self.batch_size,
                self.face_images_path,
                self.asr.feat_queue,
                self.asr.output_queue,
                self.result_queue,
            ),
        )
        self.inference_process.start()

    def _load_avatar(self):
        with open(self.coords_path, "rb") as file:
            self.coord_list_cycle = pickle.load(file)
        self.frame_list_cycle = read_images(list_image_files(self.full_images_path))

    def _get_idle_frame(self, frame_index):
        return self.frame_list_cycle[frame_index]

    def _blend_frame(self, frame_index, lip_frame):
        y1, y2, x1, x2 = self.coord_list_cycle[frame_index]
        base_frame = self.frame_list_cycle[frame_index].copy()
        try:
            resized_lip_frame = cv2.resize(lip_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        except Exception:
            return None
        base_frame[y1:y2, x1:x2] = resized_lip_frame
        return base_frame
    # 把当前这组音频数据，包装成 WebRTC 能发出去的音频帧，并塞进音频轨道队列
    def _push_audio_frames(self, audio_frames, loop, audio_track):
        for audio_data, _ in audio_frames:
            pcm_data = (audio_data * 32767).astype(np.int16)
            frame = AudioFrame(format="s16", layout="mono", samples=pcm_data.shape[0])
            frame.planes[0].update(pcm_data.tobytes())
            frame.sample_rate = 16000
            asyncio.run_coroutine_threadsafe(audio_track._queue.put(frame), loop)

    def put_msg_txt(self, msg):
        self.tts.put_msg_txt(msg)

    def put_audio_frame(self, audio_chunk):
        self.asr.put_audio_frame(audio_chunk)

    def pause_talk(self):
        self.tts.pause_talk()
        self.asr.pause_talk()
        
    # 把结果封装成音视频帧并推到 WebRTC
    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        while not quit_event.is_set():
            try:
                lip_frame, frame_index, audio_frames = self.result_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            is_idle_batch = audio_frames[0][1] != 0 and audio_frames[1][1] != 0
            if is_idle_batch:
                video_frame = self._get_idle_frame(frame_index)
            else:
                video_frame = self._blend_frame(frame_index, lip_frame)
                if video_frame is None:
                    continue

            frame = VideoFrame.from_ndarray(video_frame, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(frame), loop)
            self._push_audio_frames(audio_frames, loop, audio_track)

        print("lipreal process_frames 线程停止")

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        self.tts.render(quit_event)

        process_thread = Thread(
            target=self.process_frames,
            args=(quit_event, loop, audio_track, video_track),
        )
        process_thread.start()

        self.render_event.set()
        while not quit_event.is_set():
            self.asr.run_step()
            if video_track._queue.qsize() >= 5:
                print("sleep qsize=", video_track._queue.qsize())
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)

        self.render_event.clear()
        print("lipreal线程停止")
