# SLIMAI

SLIMAI is a comprehensive deep learning framework designed to accelerate model development across various tasks including classification, detection, and segmentation. Built with flexibility and efficiency in mind, it leverages PyTorch & MMengine library for optimized training pipelines.

## Features

### 🚀 Core Capabilities
- **Unified Training Pipeline**: Streamlined training process
- **Multi-Task Support**: Ready-to-use implementations for:
  - Classification
  - Object Detection
  - Segmentation
  - Sell Supervised Learning
  - Extensible for custom tasks

### 🛠 Development Tools
- **Modular Architecture**
  - Flexible data processing pipeline
  - Customizable model components
  - Extensible training runners
  - Rich utility functions

### 📊 Data Processing
- **Comprehensive Data Pipeline**
  - Multiple data source support
  - Rich data transformations
  - Custom dataset implementations
  - Efficient data loading

### 🔧 Training Features
- **Advanced Training Support**
  - Distributed training capability
  - Checkpoint management
  - Gradient handling
  - Model export utilities

## Usage

### Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training on single GPU:
```bash
python tools/run.py --config slimai/templates/dino.py
```

3. Run training on multiple GPU:
```bash
bash scripts/run_ddp.sh slimai/templates/dino.py
```
