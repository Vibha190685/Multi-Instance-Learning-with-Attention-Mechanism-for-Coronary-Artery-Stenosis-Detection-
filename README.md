# Multi-Instance-Learning-with-Attention-Mechanism-for-Coronary-Artery-Stenosis-Detection-

## 📌 Overview
This repository contains the implementation of **Multi-Instance Learning with Attention Mechanism for Coronary Artery Stenosis Detection on Computerized Tomography Coronary Angiography.** The goal is to detect stenosis using multi-range preprocessing and deep learning-based modeling.

## 📂 Repository Structure
Multi_Range_preprocessing.py # Implements the proposed multi-range preprocessing pipeline
modelling_MIL.py # Defines the MIL-based deep learning model and training procedure
utils_file.py # Contains all dependency functions used in modeling

**Preprocessing****: Multi-range attenuation processing for enhanced structural visualization. 
Processing Steps:
  Clipping & Normalization of each HU range
  Sobel Edge Detection for enhanced structural boundaries
  Merging of edge maps across ranges
  Thresholding & Small Object Removal to eliminate noise


**Modeling:****** MIL framework with VGG16 transfer learning, Multi-Head Attention (MHA), and positional encoding.

**Optimization:****** Hyperparameter tuning, class balancing, and data augmentation for improved model performance.





