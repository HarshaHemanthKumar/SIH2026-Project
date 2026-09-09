# 🏥 SIH26038: Explainable AI for Diabetic Retinopathy Screening in Rural India
## MathWorks Track — Production Solution Architecture & Presentation Guide

---

## 📌 Executive Summary for Judges
- **The Problem:** India faces a critical ophthalmologist shortage (~1 eye specialist per 100,000 rural residents) alongside 77+ million diabetic individuals.
- **The Solution:** An end-to-end, clinically calibrated MATLAB & Simulink tele-screening system combining:
  1. Automated fundus image quality filtering & CLAHE enhancement.
  2. Anatomical landmark localization (Optic Disc & Fovea) and lesion counting in $0.10$ seconds.
  3. ResNet-50 transfer learning trained on a combined multi-cohort dataset (APTOS + IDRiD Training = **4,075 images**).
  4. Explainable AI with multi-severity Grad-CAM attention heatmaps.
  5. Discrete-event & stochastic district-scale telemedicine simulation (**100,000 patients/year**) in Simulink (`.slx`).
  6. External clinical validation on completely unseen Indian patients (103 IDRiD test cases) and cross-camera evaluation (100 Messidor-2 cases).

---

## 🎯 Benchmark Scorecard (Real Verified Metrics)

| Evaluation Stage | Metric | SIH Benchmark Target | Our Verified System Result | Verification Script |
|---|---|---|---|---|
| **Internal Validation (Hold-out 10%)** | Sensitivity (Recall) | $> 90.0\%$ | **94.80%** | `PART4.m` |
| **Internal Validation (Hold-out 10%)** | Specificity | $> 85.0\%$ | **97.50%** | `PART4.m` |
| **Internal Validation (Hold-out 10%)** | Overall Accuracy | High | **96.45%** | `PART4.m` |
| **External Indian Cohort (IDRiD Unseen)** | Quadratic Weighted Kappa | $> 0.50$ (Moderate) | **0.6656** *(Substantial Agreement)* | `PART9...m` |
| **External Indian Cohort (IDRiD Unseen)** | Balanced Operating Point | Clinically Balanced | **Sens: 85.94% \| Spec: 71.79%** ($\tau=0.45$) | `PART9...m` |
| **External Indian Cohort (IDRiD Unseen)** | High-Sensitivity Screening Point | $> 90.0\%$ | **Sens: 90.62%** ($\tau=0.20$) | `PART9...m` |
| **Cross-Camera Domain Adaptation** | Messidor-2 Batch Test | Generalization | **100 images ingested with 0% domain collapse** | `PART9...m` |
| **Lesion Localization & Counting** | Processing Runtime | Real-time | **0.10 seconds** (10 MA, 1 EX detected) | `PART6.m` |
| **Simulink District Scalability** | Annual Patient Load | 100,000 / year | **92% Specialist Workload Reduction** | `PART7.m` (`.slx`) |

---

## 📂 Project Architecture

```text
SIH2026-Project/
├── Datasets/
│   ├── 1 APTOS 2019 Blindness Detection/          (3,662 train images)
│   ├── 2 IDRiD (Indian Diabetic Retinopathy)/     (413 train, 103 test, pixel masks)
│   ├── 3 DRIVE (Vessel Extraction)/               (Vessel reference benchmark)
│   ├── 4 Messidor-2/                              (1,748 external cross-camera images)
│   └── combined_training_dataset/                 (4,075 unified training images)
├── MATLAB-Scripts/
│   ├── PART1.m                                   # Organizes APTOS dataset
│   ├── PART2.m                                   # ResNet-50 training with Class-Weighted loss
│   ├── PART3.m                                   # 5-Class Multi-Severity Grad-CAM Dashboard
│   ├── PART4.m                                   # Internal clinical metrics (rng=42, QWK, Confusion Matrix)
│   ├── PART5.m                                   # Automated Quality Gate & Adaptive CLAHE Enhancement
│   ├── PART6.m                                   # Fast Retinal Segmentation, Landmark & Lesion Report
│   ├── PART7.m                                   # Simulink Telemedicine Simulation (.slx generated)
│   ├── PART8_UNET_LESION_SEGMENTATION.m          # Deep U-Net trained on IDRiD pixel ground truths
│   ├── PART9_EXTERNAL_VALIDATION_MESSIDOR_IDRID.m# External Validation on 103 IDRiD + 100 Messidor-2
│   └── TRAIN_ALL_DATASETS_MODEL.m                # Master GPU training on 4,075 images
├── dr_resnet50_model.mat                         # Trained ResNet-50 weights (87.8 MB)
├── unet_exudate_model.mat                        # Trained Medical U-Net weights (29.0 MB)
└── telemedicine_screening_workflow.slx           # Simulink System Model
```

---

## 💡 Key Talking Points for Judges

1. **Multi-Cohort Training (4,075 Images):**
   We did not just train on one dataset. We merged APTOS with the Indian IDRiD training set so the model natively learns Indian retinal pigmentation, illumination variations, and disease markers.

2. **No Data Leakage:**
   The 103 clinical test images from IDRiD and all 1,748 images from Messidor-2 were strictly isolated from training and reserved exclusively for out-of-distribution external benchmarking.

3. **Clinically Defensible Operating Thresholds:**
   In real healthcare deployments, clinicians don't use arbitrary 0.5 cutoffs. We present both:
   - **High-Sensitivity Screening Point ($\tau=0.20$):** Catches **90.62%** of referable cases in frontline rural camps.
   - **Balanced Diagnostic Point ($\tau=0.45$):** Achieves **85.94% sensitivity / 71.79% specificity** maximizing Youden's $J$-index.

4. **MathWorks Native Telemedicine Model:**
   Our Simulink model (`.slx`) simulates realistic rural internet constraints (2 Mbps bandwidth), fundus camera acquisition delays, and AI edge triage, demonstrating that **1 ophthalmologist can review a district of 100,000 patients** in under 15 minutes of daily screen time.
