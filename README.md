# TMCNet
**Fine-Grained Talking Motion Consistent Network for Realistic Face Generation**

TMCNet is a motion-consistent talking face generation framework designed to produce high-fidelity, temporally stable, and lip-synchronized facial animations.  
Given a source face video and an arbitrary driving audio, TMCNet generates realistic talking face videos with fine-grained motion consistency.

---

## ✨ Highlights

- 🎯 Fine-grained facial motion modeling  
- 🔊 Accurate lip synchronization with arbitrary audio  
- 🎥 Strong temporal consistency  
- 🔄 Identity-preserving generation  

---

## 🛠 Installation

We recommend using a clean Python environment (**Python 3.10+**).

```bash
git clone https://github.com/xz0305/TMCNet.git
cd TMCNet

# Create virtual environment
conda create -n tmcnet python=3.10
conda activate tmcnet

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Inference
### 📦 1. Download Pretrained Model
Download the pretrained checkpoint [tmcnet](https://drive.google.com/file/d/1pblfITK7THwjCWGWLz7ehjdoMfXYX183/view?usp=drive_link)
### ▶️ 2. Run Inference

```bash
You can lip-sync any video to any audio:
python inference.py \
    --checkpoint_path checkpoints/tmcnet.pth \
    --face inputs/source.mp4 \
    --audio inputs/speech.wav
```


## Training
### 📦 1. Dataset Preparation

TMCNet can be trained on high-quality talking face datasets such as:

- **MEAD**: https://github.com/uniBruce/Mead  
- **HDTF**: https://github.com/MRzzm/HDTF  

Please follow the official instructions to download the dataset.

### 📥 2. Download Required Pretrained Weights

Before training, download:

- ✅ Lip-expert discriminator (SyncNet) checkpoint--[syncnet.pth](https://drive.google.com/file/d/1pblfITK7THwjCWGWLz7ehjdoMfXYX183/view?usp=drive_link)
- 运动估计模型权重(https://drive.google.com/file/d/1pblfITK7THwjCWGWLz7ehjdoMfXYX183/view?usp=drive_link)

### ⚙️ 3. Data Preprocessing
```bash
python preprocess.py \
    --data_root data/ \
    --preprocessed_root MEAD_preprocessed/

Extract Facial Landmarks
python extract_lms.py \
    --data_root MEAD_preprocessed/
```

### ⚙️ 4. Running
```bash

python train.py \
    --data_root MEAD_preprocessed/ \
    --checkpoint_dir checkpoints/train_exp/ \
    --syncnet_checkpoint_path checkpoints/syncnet.pth
```
