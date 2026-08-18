import os
import random
from os.path import join

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import normalize, Crop, Flip, ToTensor


class InfraH5EventDataset(Dataset):
    def __init__(self, root, frames, future_frames, past_frames, crop_size=(256, 256), data_format='RGB',
                 centralize=True, normalize=True, event_window_size=1000):
        assert frames - future_frames - past_frames >= 1
        self.frames = frames
        self.num_ff = future_frames
        self.num_pf = past_frames
        self.data_format = data_format
        self.crop_size = crop_size
        if crop_size is None:
            self.crop_h, self.crop_w = None, None
            self.transform = transforms.Compose([Flip(), ToTensor()])
        else:
            self.crop_h, self.crop_w = crop_size
            self.transform = transforms.Compose([Crop(crop_size), Flip(), ToTensor()])
        self.normalize = normalize
        self.centralize = centralize
        self.event_window_size = event_window_size

        # 只支持h5文件
        self.root = root
        self.h5_files = sorted([f for f in os.listdir(root) if f.endswith('.h5')])

        if not self.h5_files:
            raise ValueError(f"在 {root} 目录下没有找到h5文件")

        self._samples = self._generate_samples()

        # 读取一张图片确定H,W
        with h5py.File(os.path.join(root, self.h5_files[0]), 'r') as f:
            self.H, self.W = f['images']['image000000000'].shape[:2]

    def _generate_samples(self):
        samples = []

        # 处理h5文件
        for h5_file in self.h5_files:
            h5_path = os.path.join(self.root, h5_file)
            with h5py.File(h5_path, 'r') as f:
                num_frames = len(f['images'].keys())
                for i in range(num_frames - self.frames + 1):
                    samples.append({
                        'file': h5_path,
                        'start_frame': i,
                        'end_frame': i + self.frames
                    })

        return samples

    def __getitem__(self, item):
        # 先resize到crop_size，crop参数固定为0
        top = 0
        left = 0
        flip_lr = random.randint(0, 1)
        flip_ud = random.randint(0, 1)
        sample = {'top': top, 'left': left, 'flip_lr': flip_lr, 'flip_ud': flip_ud}

        sample_data = self._samples[item]
        # 加载4个输入：模糊图像、清晰图像、事件图和事件时间戳
        blur_imgs, sharp_imgs, event_maps, event_timestamps = self._load_h5_sample(sample_data, sample)

        # 对于训练，我们需要所有帧的事件图，但只需要目标帧的清晰图像
        sharp_imgs = sharp_imgs[self.num_pf:self.frames - self.num_ff]

        # 确保所有元素都是张量并调整维度
        processed = []
        for item in [blur_imgs, sharp_imgs, event_maps, event_timestamps]:
            # 转换为张量并添加帧维度
            tensors = [torch.as_tensor(x).unsqueeze(0) for x in item]
            # 拼接成 (T, C, H, W) 格式
            processed.append(torch.cat(tensors, dim=0))

        return processed

    def _load_h5_sample(self, sample_data, sample):
        blur_imgs, sharp_imgs, event_maps, event_timestamps = [], [], [], []

        event_sample_data = self._get_event_sample_data(sample_data)
        with h5py.File(sample_data['file'], 'r') as f, \
                h5py.File(event_sample_data['file'], 'r') as event_f:
            events = event_f['events']
            images = f['images']
            sharp_images = f['sharp_images']
            event_image_key = f"image{event_sample_data['start_frame']:09d}"
            event_src_h, event_src_w = event_f['images'][event_image_key].shape[:2]

            # 获取事件数据
            ps = events['ps'][:]  # polarity
            ts = events['ts'][:]  # timestamp
            xs = events['xs'][:]  # x coordinate
            ys = events['ys'][:]  # y coordinate

            # 计算时间窗口参数（用于归一化）
            total_time = ts[-1] - ts[0] if len(ts) > 0 else 1.0

            for offset, frame_idx in enumerate(range(sample_data['start_frame'], sample_data['end_frame'])):
                # 加载图像
                img_key = f'image{frame_idx:09d}'
                img = images[img_key][:]
                label = sharp_images[img_key][:]

                # 处理图像
                if self.data_format == 'RGB':
                    img = img
                    label = label
                elif self.data_format == 'RAW':
                    img = img[..., np.newaxis].astype(np.int32)
                    label = label[..., np.newaxis].astype(np.int32)

                src_h, src_w = img.shape[:2]
                if self.crop_size is None:
                    target_size = (src_w, src_h)
                    img = img.copy()
                    label = label.copy()
                else:
                    target_size = (self.crop_w, self.crop_h)
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR).copy()
                    label = cv2.resize(label, target_size, interpolation=cv2.INTER_LINEAR).copy()

                # 生成事件图和时间戳
                event_frame_idx = event_sample_data['start_frame'] + offset
                event_map, frame_ts = self._generate_event_map_and_timestamp(
                    ps, ts, xs, ys, event_frame_idx, target_size, total_time,
                    source_size=(event_src_w, event_src_h))

                sample['image'] = img
                sample['label'] = label
                sample['event_map'] = event_map
                sample['event_timestamp'] = frame_ts  # 添加时间戳到样本
                sample = self.transform(sample)

                val_range = 2.0 ** 8 - 1 if self.data_format == 'RGB' else 2.0 ** 16 - 1
                blur_img = normalize(sample['image'], centralize=self.centralize, normalize=self.normalize,
                                     val_range=val_range)
                sharp_img = normalize(sample['label'], centralize=self.centralize, normalize=self.normalize,
                                      val_range=val_range)

                # 确保事件图和时间戳具有正确的通道维度
                event_tensor = sample['event_map']  # 应该是 (2, H, W)
                timestamp_tensor = sample['event_timestamp']  # 应该是 (1, H, W)

                blur_imgs.append(blur_img)
                sharp_imgs.append(sharp_img)
                event_maps.append(event_tensor)
                event_timestamps.append(timestamp_tensor)

        return blur_imgs, sharp_imgs, event_maps, event_timestamps

    def _generate_event_map_and_timestamp(self, ps, ts, xs, ys, frame_idx, target_size, total_time, source_size=None):
        """生成两通道事件图和对应的归一化时间戳"""
        # 创建两通道事件图
        event_map = np.zeros((target_size[::-1][0], target_size[::-1][1], 2), dtype=np.float32)  # (H, W, 2)

        # 计算时间窗口
        frame_ts = 0.0  # 默认时间戳
        if len(ts) > 0:
            # 基于时间戳的时间窗口划分
            time_per_frame = total_time / max(1, len(ts) // self.event_window_size)
            start_time = ts[0] + frame_idx * time_per_frame
            end_time = start_time + time_per_frame

            # 计算该帧事件的归一化时间戳（0-1范围）
            frame_ts = (start_time + end_time) / 2  # 使用窗口中间时间作为该帧时间戳
            frame_ts = (frame_ts - ts[0]) / total_time  # 归一化到[0, 1]范围

            # 找到时间窗口内的事件
            mask = (ts >= start_time) & (ts < end_time)
            frame_ps = ps[mask]
            frame_xs = xs[mask]
            frame_ys = ys[mask]
        else:
            # 如果没有时间戳，使用简单的事件窗口划分和帧索引作为时间戳
            start_idx = frame_idx * self.event_window_size
            end_idx = min(start_idx + self.event_window_size, len(ps))
            frame_ps = ps[start_idx:end_idx]
            frame_xs = xs[start_idx:end_idx]
            frame_ys = ys[start_idx:end_idx]

            # 使用帧索引作为时间戳并归一化
            frame_ts = frame_idx / (self.frames - 1) if self.frames > 1 else 0.5

        # 创建时间戳图（单通道，所有位置都是该帧的时间戳）
        timestamp_map = np.full((target_size[::-1][0], target_size[::-1][1], 1),
                                frame_ts, dtype=np.float32)

        if len(frame_ps) > 0:
            # 将事件坐标映射到目标尺寸
            source_w, source_h = source_size or (self.W, self.H)
            scale_x = target_size[0] / source_w
            scale_y = target_size[1] / source_h

            frame_xs = (frame_xs * scale_x).astype(int)
            frame_ys = (frame_ys * scale_y).astype(int)

            # 累积事件到对应通道
            for i in range(len(frame_ps)):
                x, y = frame_xs[i], frame_ys[i]
                if 0 <= x < target_size[0] and 0 <= y < target_size[1]:
                    if frame_ps[i] > 0:  # 正事件
                        event_map[y, x, 0] += 1
                    else:  # 负事件
                        event_map[y, x, 1] += 1

        # 对事件图进行归一化
        max_events = np.max(event_map)
        if max_events > 0:
            event_map = event_map / max_events

        return event_map, timestamp_map

    def _get_event_sample_data(self, sample_data):
        """Return the event source paired with an infrared sample.

        Ablation datasets override this hook while leaving the infrared input
        and sharp target untouched.
        """
        return sample_data

    def __len__(self):
        return len(self._samples)


class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        root = os.path.join(para.data_root, ds_type)
        frames = para.frames
        dataset = InfraH5EventDataset(root, frames, para.future_frames, para.past_frames, para.patch_size,
                                      para.data_format,
                                      para.centralize, para.normalize, getattr(para, 'event_window_size', 1000))
        gpus = para.num_gpus
        bs = para.batch_size
        ds_len = len(dataset)
        loader_kwargs = {
            'num_workers': para.threads,
            'pin_memory': True,
            'drop_last': True
        }
        if para.threads > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 2
        if para.trainer_mode == 'ddp':
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=para.num_gpus,
                rank=device_id
            )
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=False,
                sampler=sampler,
                **loader_kwargs
            )
            loader_len = np.ceil(ds_len / gpus)
            self.loader_len = int(np.ceil(loader_len / bs) * bs)
        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=True,
                **loader_kwargs
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len
