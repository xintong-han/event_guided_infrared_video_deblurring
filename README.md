# Paper Project - Event-guided Infrared Video Deblurring

The paper "Event-guided Infrared Video Deblurring" is currently under review for an IEEE journal. This paper introduces an innovative event-guided infrared video deblurring model, designed to leverage the complementary advantages of event cameras and infrared imaging to enhance visual recovery in dynamic scenes.

## Repository layout

```text
.
├── data/
│   ├── infra_h5.py       
│   └── utils.py          
├── model/
│   ├── EIVD.py           
│   ├── arches.py         
│   └── model.py          
├── test.py               
├── test.sh               
├── requirements.txt
└── README.md
```

## Checkpoint

Model weights are not contained in this repository and will be released independently. Please place the downloaded weights into the default path shown below.

Download Link：[夸克网盘](https://pan.quark.cn/s/65169fefce60?pwd=aPRa)。

Extraction code：`aPRa`

```text
weights/model_best_eivd.pth.tar
```

## Environment

The dependency files of this project record the following environment configurations:

- Python 3.8.19
- PyTorch 1.11.0
- torchvision 0.12.0
- CUDA 11.3
- Linux 或 WSL (`test.sh` is a Bash script)

We recommend creating a standalone environment with Conda:

```bash
conda create -n eivd python=3.8.19
conda activate eivd
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu113 -r requirements.txt
```

## Dataset format

Download Link：[夸克网盘](https://pan.quark.cn/s/619803fea2b4?pwd=qYGe)。

Extraction code：`qYGe`

Recommended directory structure:

```text
DATA_ROOT/
└── test/
    ├── sequence_001.h5
    ├── sequence_002.h5
    └── ...
```

Each HDF5 file must contain at least the following groups and datasets:

```text
sequence_001.h5
├── images/
│   ├── image000000000
│   ├── image000000001
│   └── ...
├── sharp_images/
│   ├── image000000000
│   ├── image000000001
│   └── ...
└── events/
    ├── ps
    ├── ts
    ├── xs
    └── ys
```


## Test

```bash
PYTHON_BIN=/path/to/conda/envs/eivd/bin/python \
DATA_ROOT=/path/to/DATA_ROOT \
CHECKPOINT=/path/to/model_best_eivd.pth.tar \
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT=results/eivd_test \
./test.sh
```





Thank you for your interest, and we look forward to advancing together with the academic and developer communities in the future!
