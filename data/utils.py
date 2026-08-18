import numpy as np
import torch


class Crop(object):
    """
    Crop randomly the image in a sample.
    Args: output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        top, left = sample['top'], sample['left']
        new_h, new_w = self.output_size
        sample['image'] = image[top: top + new_h,
                          left: left + new_w].copy()
        sample['label'] = label[top: top + new_h,
                          left: left + new_w].copy()
        
        # 处理事件图（如果存在）
        if 'event_map' in sample:
            sample['event_map'] = sample['event_map'][top: top + new_h,
                                                     left: left + new_w].copy()

        return sample


class Flip(object):
    """
    shape is (h,w,c)
    """

    def __call__(self, sample):
        flag_lr = sample['flip_lr']
        flag_ud = sample['flip_ud']
        if flag_lr == 1:
            sample['image'] = np.fliplr(sample['image']).copy()
            sample['label'] = np.fliplr(sample['label']).copy()
            if 'event_map' in sample:
                sample['event_map'] = np.fliplr(sample['event_map']).copy()
        if flag_ud == 1:
            sample['image'] = np.flipud(sample['image']).copy()
            sample['label'] = np.flipud(sample['label']).copy()
            if 'event_map' in sample:
                sample['event_map'] = np.flipud(sample['event_map']).copy()

        return sample


class Rotate(object):
    """
    shape is (h,w,c)
    """

    def __call__(self, sample):
        flag = sample['rotate']
        if flag == 1:
            sample['image'] = sample['image'].transpose(1, 0, 2)
            sample['label'] = sample['label'].transpose(1, 0, 2)

        return sample


class Sharp2Sharp(object):
    def __call__(self, sample):
        flag = sample['s2s']
        if flag < 1:
            sample['image'] = sample['label'].copy()
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # 移除 [np.newaxis, :]，避免添加多余维度
        image = np.ascontiguousarray(image.transpose((2, 0, 1)))
        label = np.ascontiguousarray(label.transpose((2, 0, 1)))
        sample['image'] = torch.from_numpy(image.copy()).float()
        sample['label'] = torch.from_numpy(label.copy()).float()

        # 处理事件图（如果存在）
        if 'event_map' in sample:
            event_map = sample['event_map']
            # 事件图是两通道的 (H, W, 2)，转换为 (2, H, W)
            # 同样移除 [np.newaxis, :]
            event_map = np.ascontiguousarray(event_map.transpose((2, 0, 1)))
            sample['event_map'] = torch.from_numpy(event_map.copy()).float()

        # 处理事件时间戳（如果存在）
        if 'event_timestamp' in sample:
            timestamp = sample['event_timestamp']
            # 时间戳是单通道的 (H, W, 1)，转换为 (1, H, W)
            timestamp = np.ascontiguousarray(timestamp.transpose((2, 0, 1)))
            sample['event_timestamp'] = torch.from_numpy(timestamp.copy()).float()

        return sample


def normalize(x, centralize=False, normalize=False, val_range=255.0):
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range

    return x


def normalize_reverse(x, centralize=False, normalize=False, val_range=255.0):
    if normalize:
        x = x * val_range
    if centralize:
        x = x + val_range / 2

    return x
