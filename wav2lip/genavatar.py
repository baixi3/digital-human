from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
import face_detection

# python genavatar.py --avatar_id='wav2lip_meilanfang' --video_path ./video/meilanfang.mp4

parser = argparse.ArgumentParser(description='为数字人生成头像帧、人脸裁剪图和坐标数据。')
parser.add_argument('--img_size', default=96, type=int)
parser.add_argument('--avatar_id', default='wav2lip_avatar1', type=str)
parser.add_argument('--video_path', default='', type=str)
parser.add_argument('--nosmooth', default=False, action='store_true',
					help='不对人脸检测结果做时间窗口平滑处理')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='人脸框扩边，顺序为 上 下 左 右，建议至少把下巴包含进去')
parser.add_argument('--face_det_batch_size', type=int, 
					help='人脸检测时的批处理大小', default=16)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'用 {device} 推理.')

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def read_imgs(img_list):
    frames = []
    print('读取图片ing ...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:  
			if batch_size == 1: 
				raise RuntimeError('图片太大GPU爆了. 调整一下 --resize_factor argument')
			batch_size //= 2
			print(f'新的 batch size: {batch_size}')
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # 查看这个未检测到人脸的帧。
			raise ValueError('未检测到人脸，请确保视频中的每一帧都包含人脸。')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

if __name__ == "__main__":
    avatar_path = f"./results/avatars/{args.avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    osmakedirs([avatar_path,full_imgs_path,face_imgs_path])
    print(args)

    #if os.path.isfile(args.video_path):
    video2imgs(args.video_path, full_imgs_path, ext = 'png') # 把视频拆成图片
    input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

    frames = read_imgs(input_img_list) # 读取图片
    face_det_results = face_detect(frames) # 人脸检测结果，人脸框和坐标
    coord_list = []
    idx = 0
    for frame, coords in face_det_results:        
        #x1, y1, x2, y2 = bbox
        resized_crop_frame = cv2.resize(frame,(args.img_size, args.img_size)) # 裁剪出人脸
        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
        coord_list.append(coords)
        idx = idx + 1
	
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f) 
