# CorridorKey 2.0 Pro GUI

> **Professional Studio Dark Desktop Interface for CorridorKey 2.0 AI Video Keyer with Native SAM 3.1 Tracking, BiRefNet Matting, and GPU Acceleration.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20x64-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.4%2B-green.svg)]()

---

## 🌟 Overview

**CorridorKey 2.0 Pro GUI** brings the cutting-edge raw-resolution video matting capabilities of **CorridorKey 2.0** into a standalone, studio-grade desktop application.

Features include:
- **CorridorKey 2.0 Native Engine**: Full-resolution raw-video green & blue screen inference with temporally coherent alpha mattes.
- **SAM 3.1 Video Tracker**: Interactive multi-point (`1`/`2`) and bounding box (`R`) segmenter with high-speed video tracking.
- **BiRefNet Studio Matting**: Zero-annotation automatic alpha generation supporting multiple models (General, Matting, HR, Dynamic, Lite).
- **GPU Accelerated & VRAM Optimized**: Dynamic layer offloading to prevent out-of-memory errors on consumer GPUs (RTX 3060 to RTX 4090+).
- **Chroma Key & Despill**: Real-time color-difference keyer with morphological cleanup and spill suppression.
- **Dual Viewer Workspace**: Real-time side-by-side or split wipe comparison between Source, Alpha Matte, Cutout, and Checkerboard composite.

---

## 🚀 Quick Start (1-Click Portable Installation)

No global Python or external compilers required. The setup is 100% self-contained in this directory.

### 1. Clone the Repository
```bash
git clone https://github.com/tahakhorsand/CorridorKey-2.0-Pro-GUI.git
cd CorridorKey-2.0-Pro-GUI
```

### 2. Run the Installer
Double-click:
```text
install.bat
```
This automatically fetches an isolated portable CPython 3.12 environment, creates `backend/venv`, and installs PyTorch with CUDA acceleration along with all Python dependencies. **No heavy model files are downloaded during this step.**

### 3. Launch the Application
Double-click:
```text
Run_GUI.bat
```
On first launch, the built-in **Download Manager** will automatically appear, allowing you to download the essential model weights and FFmpeg directly with resumable download support.

---

## 📁 Clean Workspace Layout

Once set up, the root folder contains only what you need:
```text
CorridorKey-2.0-Pro-GUI/
├── backend/          # Core services, UI, models, runtime, and embedded environment
├── logs/             # Session execution logs
├── Projects/         # User video projects workspace and extracted frames
├── Run_GUI.bat       # 1-Click GUI launcher
├── install.bat       # 1-Click portable environment installer
├── LICENSE           # Project license (MIT)
└── README.md         # Documentation
```

---

## 📜 Attribution, Forks & Lineage

- **CorridorKey 2.0**: This project is a specialized fork and user-interface development built upon **[CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0)** created by **Niko Pueringer / Corridor Digital**.
- **EZ-Corridor Inspiration**: Architectural patterns, desktop Qt workflows, and UI widgets are inspired by and adapted from the open-source **EZ-Corridor** project.
- **Development**: Architecture redesign, folder modularization, SAM 3.1 native integration, and Pro Studio Dark UI enhancements developed by **Taha Khorsand (@tahakhorsand)**.

---

## ⚖️ Third-Party Models & Licensing Disclaimer

This software code is licensed under the **MIT License**. The underlying machine learning model checkpoints, vision foundation models, and external binaries used by CorridorKey 2.0 are published under their own respective licenses by their original creators. These models are not hosted in this Git repository and are fetched directly from their official public repositories:

| Model / Asset | Original Publisher / Author | License | Source / Upstream Repository |
| :--- | :--- | :--- | :--- |
| **CorridorKey 2.0 Weights** | Corridor Digital (Niko Pueringer) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | [HuggingFace: nikopueringer/CorridorKey2.0](https://huggingface.co/nikopueringer/CorridorKey2.0) |
| **C-RADIOv4-SO400M** | NVIDIA Corporation | [NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license.pdf) | [HuggingFace: nvidia/C-RADIOv4-SO400M](https://huggingface.co/nvidia/C-RADIOv4-SO400M) |
| **MoGe-2 ViT-B** | Ruicheng Wang / Microsoft | [MIT License](https://opensource.org/licenses/MIT) | [HuggingFace: Ruicheng/moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) |
| **RVM (Robust Video Matting)** | Google DeepMind / PeterL1n | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [Google Cloud Storage / PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) |
| **SAM 3.1 Tracking Weights** | Open Source Community / Meta | Apache 2.0 / Open Community Model | Public Safetensors HuggingFace Repositories |
| **BiRefNet** | ZhengPeng7 | [MIT License](https://opensource.org/licenses/MIT) | [HuggingFace: ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) |
| **FFmpeg** | FFmpeg Developers (GyanD Builds) | [GPL v3 / LGPL v3](https://ffmpeg.org/legal.html) | [GyanD/codexffmpeg Releases](https://github.com/GyanD/codexffmpeg) |

> [!IMPORTANT]
> **CorridorKey 2.0 Pro GUI** is an independent graphical client. Users must comply with the individual licenses of any pretrained weights and models downloaded through the application.
