# PATFOM: A General-purpose Photoacoustic Tomography Foundation Model

[![Paper](https://img.shields.io/badge/Paper-PATFOM-blue)]()
[![License](https://img.shields.io/badge/License-Academic-green)]()

## 📌 Overview

This repository provides the official implementation of:

**"PATFOM: A General-purpose Photoacoustic Tomography Foundation Model for Multi-Device, Multi-Task and Multi-Organ Image Analysis"**

Photoacoustic tomography (PAT) has shown great potential for biomedical imaging, yet existing AI-based approaches are typically task-specific and lack generalizability across devices, tasks, and anatomical regions.  

To address these limitations, we propose **PATFOM**, the first foundation model for PAT imaging, designed to enable **multi-device, multi-task, and multi-organ image analysis**.

---

## 🧠 Key Features

- 🔹 First foundation model for preclinical PAT imaging  
- 🔹 Supports multiple downstream tasks:
  - Image recovery
  - Multi-organ segmentation
  - Light fluence estimation
  - Vessel enhancement  
- 🔹 Strong cross-device generalization  
- 🔹 Self-supervised dual-domain pretraining:
  - Spatial masking
  - Signal-domain masking  
- 🔹 Built on the large-scale **CMM-PAT dataset (~944K images)**  

---

## 📄 Abstract

The integration of photoacoustic tomography (PAT) imaging and artificial intelligence has accelerated the analysis of images. However, existing task-specific approaches have limited applicability across imaging devices, tasks, and anatomical regions. Building a foundation model for PAT may solve this problem, yet it remains challenging due to the lack of large-scale datasets, as well as the difficulty in learning generalizable image representations.  

Here, we propose a general-purpose PAT foundation model (**PATFOM**) designed to enhance adaptability and scalability across multi-device, multi-task, and multi-organ PAT image analysis. Central to this work is **CMM-PAT**, the largest and most comprehensive PAT dataset to date, encompassing 944,614 images across image-signal domains, diverse organs, and multiple wavelengths.  

PATFOM adopts a **dual-domain masked modeling strategy** to enable robust self-supervised pretraining. With task-specific fine-tuning, it achieves strong performance across multiple downstream tasks. Extensive experiments on three commercial PAT systems further demonstrate its robustness and generalizability.

---

## ⚙️ Requirements

- Python >= 3.8  
- PyTorch >= 1.7  
- torchvision >= 0.8  


## 📂 Dataset & Pretrained Weights

Test datasets and pretrained weights can be downloaded from:
👉 [(https://pan.baidu.com/s/1vbqeUWwCL1PJUIN4g2uCJQ?pwd=zh32)]

Due to the large scale of the pretraining dataset, full data is not publicly released.
If needed, please contact us.

📧 Contact: ytzhong.smu@qq.com

## 🚀 Usage

1️⃣ Pretraining
```bash
python scripts/PATpretrain.py \
--src_dir /public/zhongyutian/PATFOM-main/PAT_Recon_s4 \
--data_dir /public/zhongyutian/PATFOM-main/PAT_Recon_s4/imgs/ \
--save_dir ./output_Recon/ \
--b 1 --dataset ACDC --fold 0 --tr_size 1 --num_classes 1
```

```bash
python scripts/PATpretest.py \
--src_dir /public/zhongyutian/PATFOM-main/PAT_Recon_s4 \
--data_dir /public/zhongyutian/PATFOM-main/PAT_Recon_s4/imgs/ \
--save_dir ./output_Recon/ \
--b 1 --dataset ACDC --fold 0 --tr_size 1 --num_classes 1
```

2️⃣ Image Recovery
```bash
python scripts/PATrecontrain.py \
--src_dir /public/zhongyutian/PATFOM-main/PAT_Recon \
--data_dir /public/zhongyutian/PATFOM-main/PAT_Recon/imgs/ \
--save_dir ./output_Recon/ \
--b 1 --dataset ACDC --fold 0 --tr_size 1 --num_classes 1
```
```bash
python scripts/PATrecontest.py \
--src_dir /public/zhongyutian/PATFOM-main/PAT_Recon \
--data_dir /public/zhongyutian/PATFOM-main/PAT_Recon/imgs/ \
--save_dir ./output_Recon/ \
--b 1 --dataset ACDC --fold 0 --tr_size 1 --num_classes 1
```

3️⃣ Light Fluence Estimation
```bash
python scripts/PATvasculartrain.py \
--src_dir /public/zhongyutian/PATFOM-main/PAT_Peitai_IV \
--data_dir /public/zhongyutian/PATFOM-main/PAT_Peitai_IV/imgs/ \
--save_dir ./ACDC_output_IV/ \
--b 1 --dataset ACDC --fold 0 --tr_size 1 --num_classes 1
```
```bash
python scripts/PATvasculartest.py \
--src_dir /public/zhongyutian/PATFOM-main/PAT_Peitai_IV \
--data_dir /public/zhongyutian/PATFOM-main/PAT_Peitai_IV/imgs/ \
--save_dir ./ACDC_output_IV/ \
--b 1 --dataset ACDC --fold 0 --tr_size 1 --num_classes 1
```

4️⃣ Multidimensional segmentation
```bash
python scripts/PATsegtrain.py \
--src_dir /public/zhongyutian/PATFOM-main/PAT_Segmentation \
--data_dir /public/zhongyutian/PATFOM-main/PAT_Segmentation/imgs/ \
--save_dir ./ACDC_output_Sge/ \
--b 1 --dataset ACDC --fold 0 --tr_size 1 --num_classes 1
```
```bash
python scripts/PATSegtest.py
```

## 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@article{PATFOM2025,
  title={PATFOM: A General-purpose Photoacoustic Tomography Foundation Model for Multi-Device, Multi-Task and Multi-Organ Image Analysis},
  author={Zhong, Yutian and et al.},
  journal={Under Review},
  year={2025}
}
```
🚧 The manuscript is currently under peer review. A formal citation will be provided upon publication.

## 📂 📬 Contact

For questions, collaborations, or dataset access:

📧 Email: ytzhong.smu@qq.com

## ⭐ Acknowledgements

We thank all contributors and collaborators involved in building the CMM-PAT dataset and developing PATFOM.
