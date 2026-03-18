# DFSPAIM

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 📖 Overview

Dermatofibrosarcoma protuberans (DFSP) and benign fibrous histiocytoma (BFH) are common skin spindle cell tumors but are often misdiagnosed, particularly by primary care physicians. DFSP's fibrosarcomatous and myxoid variants are frequently underdiagnosed, leading to delayed or over-treatment. This study aims to develop a deep learning system to improve the histopathologic diagnosis of DFSP, BFH, and malignant fibrous histiocytoma (undifferentiated pleomorphic sarcoma, UPS).

Surgical specimens were reviewed by 2-3 subspecialists. Four computer vision models—DenseNet121, EfficientNet-B3, ResNet34, and Xception—were developed. Their performances were evaluated using confusion matrices, ROC curves, and heatmaps. The optimal model's efficiency was compared with that of nine pathologists.

A total of 131,952 patches from 639 whole slide images (WSIs) of 463 cases were analyzed. The model, DFSPAIM, achieved an AUC of 0.90 across five categories in the external validation cohort, outperforming pathologists in agreement rates at both patch and WSI levels. This study demonstrates strong performance in reducing misdiagnoses of BFH as DFSP and minimizing the risk of missing sarcomatous or myxoid variants of DFSP.


## 🏗️ Architecture

DFSPAIM utilizes four state-of-the-art computer vision models:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: WSI Patches                        │
│                     (256x256 / 512x512)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backbone Networks                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ DenseNet121  │  │ EfficientNet │  │  ResNet34    │       │
│  │              │  │    -B3       │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐                                           │
│  │  Xception    │                                           │
│  │              │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Prediction Heads (5 classes)                 │
│  Classical DFSP │ FS-DFSP │ Myxoid DFSP │ BFH │ UPS         │
└─────────────────────────────────────────────────────────────┘
```

### Classification Categories (5 Classes)

| Index | Code | Full Name |
|-------------|------|-----------|
| 0 | CD | Classical Dermatofibrosarcoma Protuberans |
| 1 | FD | Fibrosarcomatous DFSP |
| 2 | MD | Myxoid Dermatofibrosarcoma Protuberans |
| 3 | BFH | Benign Fibrous Histiocytoma |
| 4 | UPS | Undifferentiated Pleomorphic Sarcoma |

## 📊 Results

| Metric | Value |
|--------|-------|
| **Dataset Size** | 131,952 patches from 639 WSIs (463 cases) |
| **External Validation AUC** | 0.90 (5-class classification) |
| **Pathologist Agreement Rate** | Higher agreement compared to 9 pathologists |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Zaneph1/DFSPAIM.git
cd DFSPAIM

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

After downloading the dataset from https://github.com/Zaneph1/DFSP-WSI, organize it as follows:

```
data/
├── jpeg-dfsp-256x256/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── ke/
│   │   ├── image2.jpg
│   │   └── ...
│   ├── train.csv
│   └── test.csv
└── jpeg-dfsp-512x512/
    └── ...
```

### Requirements

- Python 3.6+
- PyTorch 1.7+
- CUDA 10.2+ (recommended)
- See `requirements.txt` for full list of dependencies

## 📊 Dataset

The DFSP-WSI dataset is available on GitHub:

**🔗 https://github.com/Zaneph1/DFSP-WSI**

### Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total Cases | 463 |
| Whole Slide Images (WSIs) | 639 |
| Image Patches | 131,952 |

### Data Format

- **Input**: Whole Slide Images (WSI) patches
- **Patch Sizes**: 256x256 or 512x512 pixels
- **Formats**: JPEG images with CSV annotations

### Training

Train a model using the following command:

```bash
# Train EfficientNet-B3 (5-class classification)
python train.py \
    --kernel-type dfsp_effnet_b3_256 \
    --data-dir ./data/ \
    --data-folder 256 \
    --image-size 256 \
    --enet-type efficientnet_b3 \
    --out-dim 5 \
    --n-epochs 4 \
    --use-meta \
    --CUDA_VISIBLE_DEVICES 0,1
```

### Prediction

Make predictions on test data:

```bash
python predict.py \
    --kernel-type dfsp_effnet_b3_256 \
    --data-dir ./data/ \
    --data-folder 256 \
    --image-size 256 \
    --enet-type efficientnet_b3 \
    --model-dir ./weight/eff \
    --eval best
```

### Grad-CAM Visualization

Generate interpretability heatmaps:

```bash
python hot.py
```

This will generate heatmaps showing which regions of the image the model focuses on for its predictions.

## 📁 Project Structure

```
DFSPAIM/
├── models.py           # Model definitions (4 architectures)
├── dataset.py          # Dataset loading and preprocessing
├── train.py            # Training script
├── predict.py          # Prediction/inference script
├── hot.py              # Grad-CAM heatmap visualization
├── build_dataset.py    # Build dataset CSV files
├── split_dataset.py    # Train/test split utility
├── util.py             # Utility functions (learning rate schedulers)
├── config.json         # Configuration file (paths)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 📋 Configuration

For local testing, you can edit `config.json` to set your data paths:

```json
{
  "root_folder": "data/your_images/",
  "resume_path": "weight/your_model.pth",
  "single_img_path": "data/model_pic/",
  "save_path": "output/"
}
```

## 🧠 Model Zoo

| Model | Input Size | Classes | Pretrained Weights |
|-------|------------|---------|--------------------|
| DenseNet121 | 256x256 | 5 | [PyTorch](https://download.pytorch.org/models/densenet121-a639ec97.pth) |
| EfficientNet-B3 | 256x256 | 5 | [timm](https://github.com/rwightman/pytorch-image-models) |
| ResNet34 | 256x256 | 5 | [PyTorch](https://download.pytorch.org/models/resnet34-333f7ec4.pth) |
| Xception | 256x256 | 5 | [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) |

> **Note:** All models use ImageNet pretrained weights. Train on your own dataset for 5-class classification: Classical DFSP, FS-DFSP, Myxoid DFSP, BFH, UPS

## 🔗 Related Resources

- **Dataset**: [DFSP-WSI Dataset](https://github.com/Zaneph1/DFSP-WSI)
- **Project Page**: [DFSPAIM GitHub](https://github.com/Zaneph1/DFSPAIM)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please open an issue or contact the authors directly.
