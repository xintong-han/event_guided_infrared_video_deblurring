#!/usr/bin/env python3
"""Generic test-only runner for EIVDeblur-compatible sequence datasets."""

import argparse
import hashlib
import importlib
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.utils import normalize_reverse  # noqa: E402
from model import Model  # noqa: E402


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, message=""):
        line = f"{time.strftime('%Y/%m/%d, %H:%M:%S')} - {message}"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_size(height, width):
    if height == 0 and width == 0:
        return None
    if height <= 0 or width <= 0:
        raise ValueError("--height and --width must both be positive, or both be 0")
    return height, width


def resolve_dataset_root(data_root, split):
    root = Path(data_root).expanduser().resolve()
    split_root = root / split if split else root
    if split and split_root.is_dir():
        root = split_root
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return root


def load_dataset_class(module_name, class_name):
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as error:
        raise ImportError(f"{class_name!r} was not found in {module_name!r}") from error


def build_test_dataset(base_class):
    class DeterministicTestDataset(base_class):
        """Disable random transforms while retaining the selected loader."""

        def __getitem__(self, item):
            transform_args = {"top": 0, "left": 0, "flip_lr": 0, "flip_ud": 0}
            sample_data = self._samples[item]
            loaded = self._load_h5_sample(sample_data, transform_args)
            if not isinstance(loaded, (list, tuple)) or len(loaded) < 2:
                raise RuntimeError(
                    "Dataset _load_h5_sample() must return blur, sharp and optional auxiliary sequences"
                )

            loaded = list(loaded)
            loaded[1] = loaded[1][self.num_pf : self.frames - self.num_ff]
            result = []
            for sequence in loaded:
                tensors = [torch.as_tensor(value).unsqueeze(0) for value in sequence]
                result.append(torch.cat(tensors, dim=0))
            return result

    return DeterministicTestDataset


def to_image(tensor, data_format, centralize, normalize, value_range):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if data_format == "RGB" and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_reverse(
        image,
        centralize=centralize,
        normalize=normalize,
        val_range=value_range,
    )
    image = np.clip(image, 0, value_range)
    if value_range <= 255:
        return image.astype(np.uint8)
    return image.astype(np.uint16)


def calculate_psnr(output, target, value_range):
    difference = (
        output.astype(np.float64) - target.astype(np.float64)
    ) / value_range
    mse = np.mean(difference ** 2)
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


def calculate_ssim(output, target, value_range):
    multichannel = output.ndim == 3 and output.shape[-1] > 1
    if not multichannel and output.ndim == 3:
        output = output[..., 0]
        target = target[..., 0]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`multichannel` is a deprecated argument name.*",
            category=FutureWarning,
        )
        return structural_similarity(
            target,
            output,
            multichannel=multichannel,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=value_range,
        )


def parse_target_indices(specification, output_count):
    value = specification.strip().lower()
    if value == "all":
        return list(range(output_count))
    if value == "center":
        return [output_count // 2]
    try:
        index = int(value)
    except ValueError as error:
        raise ValueError("--target-index must be 'all', 'center', or an integer") from error
    if index < 0:
        index += output_count
    if index < 0 or index >= output_count:
        raise IndexError(
            f"Target index {specification!r} is invalid for {output_count} output frames"
        )
    return [index]


def load_checkpoint(model, checkpoint_path, checkpoint_key, strict):
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=lambda storage, location: storage.cuda(),
    )
    if checkpoint_key and isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
    else:
        state_dict = checkpoint

    keys = list(state_dict.keys())
    has_module_prefix = bool(keys) and all(key.startswith("module.") for key in keys)
    if has_module_prefix:
        model = nn.DataParallel(model)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return model, checkpoint, incompatible


def sample_identity(dataset, batch_index):
    if hasattr(dataset, "_samples") and batch_index < len(dataset._samples):
        sample = dataset._samples[batch_index]
        file_path = Path(sample.get("file", f"sample_{batch_index:08d}"))
        start_frame = sample.get("start_frame", batch_index)
        return file_path.stem, start_frame
    return "samples", batch_index


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--split",
        default="test",
        help="Use DATA_ROOT/SPLIT when that directory exists; use an empty string for DATA_ROOT itself",
    )
    parser.add_argument("--dataset-name", default="infra_h5")
    parser.add_argument("--dataset-module", default="data.infra_h5")
    parser.add_argument("--dataset-class", default="InfraH5EventDataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-key", default="state_dict")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--model", default="EIVD")
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-blocks", type=int, default=15)
    parser.add_argument("--activation", default="gelu")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--past-frames", type=int, default=2)
    parser.add_argument("--future-frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=0, help="0 with --width 0 preserves native size")
    parser.add_argument("--width", type=int, default=0, help="0 with --height 0 preserves native size")
    parser.add_argument("--data-format", choices=["RGB", "RAW"], default="RGB")
    parser.add_argument("--event-window-size", type=int, default=1000)
    parser.add_argument(
        "--target-index",
        default="all",
        help="Evaluate 'all', 'center', or one integer index in the aligned output/GT sequence",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results/test"))
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--no-centralize", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required by the supplied EIVD implementation")

    crop_size = parse_size(args.height, args.width)
    dataset_root = resolve_dataset_root(args.data_root, args.split)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    output_root = Path(args.output_root).expanduser().resolve()
    logger = Logger(output_root / "log.txt")
    logger("Generic EIVDeblur test")
    for name in sorted(vars(args)):
        logger(f"{name}: {getattr(args, name)}")
    logger(f"resolved_data_root: {dataset_root}")
    logger(f"checkpoint_sha256: {sha256(checkpoint_path)}")

    centralize = not args.no_centralize
    normalize = not args.no_normalize
    value_range = 255.0 if args.data_format == "RGB" else 65535.0

    parameters = SimpleNamespace(
        activation=args.activation,
        n_features=args.n_features,
        n_blocks=args.n_blocks,
        future_frames=args.future_frames,
        past_frames=args.past_frames,
        frames=args.frames,
        profile_H=args.height or 360,
        profile_W=args.width or 640,
        model=args.model,
    )
    torch.manual_seed(39)
    torch.cuda.manual_seed_all(39)

    base_dataset_class = load_dataset_class(args.dataset_module, args.dataset_class)
    dataset_class = build_test_dataset(base_dataset_class)
    dataset = dataset_class(
        root=str(dataset_root),
        frames=args.frames,
        future_frames=args.future_frames,
        past_frames=args.past_frames,
        crop_size=crop_size,
        data_format=args.data_format,
        centralize=centralize,
        normalize=normalize,
        event_window_size=args.event_window_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    logger(f"Dataset windows: {len(dataset)}")

    model = Model(parameters).cuda()
    model, checkpoint, incompatible = load_checkpoint(
        model,
        checkpoint_path,
        args.checkpoint_key,
        strict=not args.no_strict,
    )
    model.eval()
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    logger(f"Checkpoint epoch: {epoch}")
    logger(f"Checkpoint load result: {incompatible}")

    psnr_values = []
    ssim_values = []
    inference_times = []
    processed_windows = 0
    evaluated_frames = 0
    suffix = "png" if args.data_format == "RGB" else "tiff"

    progress_total = len(dataloader)
    if args.max_samples > 0:
        progress_total = min(progress_total, args.max_samples)
    progress = tqdm(
        total=progress_total,
        desc=f"Testing {args.dataset_name}",
        unit="window",
        dynamic_ncols=True,
        disable=args.no_progress,
        file=sys.stdout,
    )

    with progress, torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            inputs = [value.cuda() for value in batch]
            blur_imgs, sharp_imgs = inputs[0], inputs[1]

            started = time.time()
            output_sequence = model(inputs)
            if isinstance(output_sequence, (list, tuple)):
                output_sequence = output_sequence[0]
            inference_times.append(time.time() - started)

            aligned_count = min(sharp_imgs.shape[1], output_sequence.shape[1])
            target_indices = parse_target_indices(args.target_index, aligned_count)
            h5_name, start_frame = sample_identity(dataset, batch_index)

            for target_index in target_indices:
                ground_truth = to_image(
                    sharp_imgs[0, target_index],
                    args.data_format,
                    centralize,
                    normalize,
                    value_range,
                )
                output = to_image(
                    output_sequence[0, target_index],
                    args.data_format,
                    centralize,
                    normalize,
                    value_range,
                )
                psnr_values.append(calculate_psnr(output, ground_truth, value_range))
                ssim_values.append(calculate_ssim(output, ground_truth, value_range))
                evaluated_frames += 1

                if args.save_images or args.save_inputs:
                    save_dir = output_root / h5_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    if len(target_indices) == 1:
                        stem = f"{batch_index:08d}"
                    else:
                        stem = f"{batch_index:08d}_f{target_index:03d}"
                    if args.save_images:
                        cv2.imwrite(str(save_dir / f"{stem}_gt.{suffix}"), ground_truth)
                        cv2.imwrite(str(save_dir / f"{stem}_output.{suffix}"), output)
                    if args.save_inputs:
                        input_index = min(
                            args.past_frames + target_index,
                            blur_imgs.shape[1] - 1,
                        )
                        input_image = to_image(
                            blur_imgs[0, input_index],
                            args.data_format,
                            centralize,
                            normalize,
                            value_range,
                        )
                        cv2.imwrite(str(save_dir / f"{stem}_input.{suffix}"), input_image)

            processed_windows += 1
            progress.update(1)
            if args.max_samples > 0 and processed_windows >= args.max_samples:
                break

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))
    mean_time = float(np.mean(inference_times))
    summary = [
        f"Test windows: {processed_windows}",
        f"Evaluated frames: {evaluated_frames}",
        f"PSNR: {mean_psnr:.10f}",
        f"SSIM: {mean_ssim:.10f}",
        f"Average inference time: {mean_time:.6f}s/window",
        f"Results: {output_root}",
    ]
    logger("")
    for line in summary:
        logger(line)
    print("\n" + "\n".join(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
