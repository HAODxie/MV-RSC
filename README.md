
# MV-RSC: Multi-View Radiographic Analysis Framework

[![License](https://img.shields.io/badge/License-MIT-%230B0B0B?style=flat-square&labelColor=white&logoWidth=20)](https://github.com/HAODxie/MV-RSC/blob/main/LICENSE)


## Dataset Structure

```
data/
└── plaques/
    ├── test/
    │   ├── dataset1/
    │   │   ├── VP/
    │   │   └── CP/
    │   ├── dataset2/
    │   │   ├── VP/
    │   │   └── CP/
    │   └── ... (dataset3-dataset9)
    └── train/
        ├── dataset1/
        │   ├── VP/
        │   └── CP/
        ├── dataset2/
        │   ├── VP/
        │   └── CP/
        └── ... (dataset3-dataset9)
```

- **Hierarchy**:
  - Root: `data`
  - Level 1: `plaques`
  - Level 2: `test`/`train`
  - Level 3: `dataset1-dataset9`
  - Level 4: `VP`  & `CP` 

- **Image Formats**: PNG, JPG, JPEG (224×224、Grayscale images recommended)

## Installation

```bash
git clone https://github.com/HAODxie/MV-RSC.git
cd MV-RSC
```

## Training Pipeline

1. **Prepare Dataset**
   - Maintain directory structure as shown above
   - Ensure balanced image distribution between VP/CP folders

2. **Base Models Training**
```bash
python RSC_train.py
```
   - Generates 9 models in `runs/train_RSCNN_Parallel/data1-9/fold_[1-5]`
   - Each fold contains:
     - `best.pt`: Optimal weights
     - `last.pt`: Final weights

3. **Fusion Model Training**
```bash
python MV_RSC_train.py
```
   - Uses `fold_1/best.pt` as base
   - Outputs to `runs/train_RSCNN_Parallel/train_attention`

## Evaluation

| Model Type | Command                   | Output Directory     |
|------------|---------------------------|----------------------|
| Base       | `python RSC_detect.py`    | `runs/detect/detect_RSCNN_Parallel`    |
| Fusion     | `python MV_RSC_detect.py` | `runs/detect/detect_RSCNN_Parallel/train_attention` |

## Output Structure



## Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.3+ (recommended)
- 16GB+ VRAM for optimal performance

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Simply copy this entire content into a new file named `README.md` in your project root directory. The Markdown formatting will render properly on GitHub.
