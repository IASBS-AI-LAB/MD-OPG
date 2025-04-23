# Detection of Pediatric Dental Caries in Panoramic Radiographs Using Artificial Intelligence: A Benchmark Study on MD-OPG

**Authors**: *Hadi Rahimi*, *Mohammad-rasoul Naeimi*, *Dr. Parvin Razzaghi*, *Dr. Bahare Nazemi*, *Dr. Shayan Darvish*
**Affiliation**: Institute for Advanced Studies in Basic Sciences (IASBS), Zanjan University of Medical Sciences (ZUMS), University of Michigan  
**Contact**: [Hadirahimi@iasbs.ac.ir, hadi.rahimi7171@gmail.com] ,[p.razzaghi@iasbs.ac.ir] , 


---

## 📌 Overview

In this study, we introduce a Panoramic Radiographs dataset **(MD-OPG)** for Pediatric Dental Caries detection and provide benchmarking experiments for both **classification** and **segmentation** tasks.

We evaluate the performance of standard deep learning models using various loss functions, providing a solid baseline for future research.

---

## 📁 Repository Structure

```bash
classification/      # Code for classification task using pretrained ResNet-18 (Keras) + patch extraction code for training this model
segmentation/        # Code for segmentation using UNet and Attention-UNet (Torch) + extracting smile zone images code for these models
figures/             # Supplementary plots (loss and Dice score plots per 3 loss experiments for both segmentation models)
README.md

---
