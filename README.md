# LightIDS: A Lightweight Neural Network-based Intrusion Detection System

**Authors:** Ebrahim Fard · Mahdi Soltani · Amir Hossein Jahangir · Seokbum Ko  

---

## Overview
**LightIDS** is a Deep Neural Network-based Intrusion Detection System (IDS) designed for **resource-constrained environments**.  
Its lightweight architecture enables efficient deployment while maintaining high accuracy in network threat detection.  

This repository contains the source code and implementation details for the paper:  
*“LightIDS: A Lightweight Neural Network-based Intrusion Detection System.”*

---

## Datasets
We evaluated **LightIDS** using two benchmark datasets:

- [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html)

For additional background and reference implementations, see:
- [Related paper on SpringerLink](https://link.springer.com/article/10.1007/s10207-021-00567-2)
- [Continual-Federated-IDS repository](https://github.com/INL-Laboratory/Continual-Federated-IDS)

---

## LightIDS Model

We trained and tested the model on **Ubuntu 24.04**.  
All required packages are listed in the `requirements.txt` file.  
It is highly recommended to use a Python virtual environment before running the code.

By running `LightIDS_main.py`, you should see:

![Screenshot from 2024-07-14 20-03-54](https://github.com/user-attachments/assets/c890e423-0f08-43a8-9c0f-edcd7b27f500)


And upon completion:

![Screenshot from 2024-07-14 20-00-03](https://github.com/user-attachments/assets/c1badf61-1435-4ff7-8194-93541d032cc7)


Note: These are sample outputs — results may vary depending on dataset size and preprocessing configuration.

## Results
### ROC Curve (Receiver Operating Characteristic)

The ROC curve evaluates the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR).
A curve closer to the top-left corner indicates stronger discrimination ability.

Here, the ROC (Receiver Operating Characteristic) curve and AUC of the model show its impactful performance on [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html).

<img width="425" height="340" alt="roc_curve" src="https://github.com/user-attachments/assets/7531ff14-64cf-42b3-a167-23f965d3e317" />

### Precision–Recall (PR) Curve

The PR curve is especially useful for imbalanced datasets, showing the relationship between precision (accuracy of positive predictions) and recall (coverage of actual positives).

A higher and more stable curve indicates that the model maintains high precision even at high recall levels.

**AUPRC (Area Under the Precision–Recall Curve) = 0.962**, demonstrating strong discriminative capability and robustness against class imbalance.

<img width="425" height="340" alt="pr_curve" src="https://github.com/user-attachments/assets/ccf2e869-1836-4f4c-8188-268fd517b26c" />

### Calibration Metrics

Calibration metrics measure how well the model’s predicted probabilities reflect real-world likelihoods.

A **Brier Score of 0.048** confirms that the predicted probabilities are well-aligned with actual outcomes.
A low score signifies strong calibration and trustworthy probability outputs, even under the 81.8% benign traffic imbalance in the dataset.

<img width="425" height="340" alt="calibration_curve" src="https://github.com/user-attachments/assets/a89934c1-1e58-4ee2-a447-f43446e19dc3" />

### Leakage-Safe Splitting

The above results are delivered by running the `LightIDS_main.py` file. To prevent data leakage in time-dependent datasets, a leakage-safe temporal split can be performed via the `LightIDS_leaksafe.py` file. However, due to the dataset’s natural imbalance, this method is expected to produce lower metrics, for example:

- Precision = 66.67%
- Recall = 22.86%
- F1-Score = 34.04%

## Hardware Implementation
LightIDS can be deployed on the [AMD ZC706 Evaluation Board](https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/ek-z7-zc706-g.html) using the configuration files:

- `design_1_wrapper.bit`
- `design_1.hwh`

Support for other FPGA platforms, such as [PYNQ-Z2](https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html), will be added later.

### System Architecture
The overall hardware architecture of LightIDS on the ZC706 Zynq SoC is shown below:

<img width="515" height="345" alt="fig_hw_acc" src="https://github.com/user-attachments/assets/f51bded6-6ede-4d66-831f-615376b7a63b" />

### Resource Utilization
The following figure presents the hardware resource utilization of the implemented design, generated using [Vivado IDE](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado.html).

<img width="810" height="270" alt="Screenshot From 2025-10-19 22-43-05" src="https://github.com/user-attachments/assets/06d772ae-a8b8-41c8-9e3d-2d2797289378" />

If you find this project useful, please consider starring the repository to support the research.
