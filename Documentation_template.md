# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Neural Nexus
**Team Members:** Himanshu(Team Leader), Vaibhav, Jigisha Saigal, Yashika Moni
**Submission Date:** 12-10-2025

---

## 1. Executive Summary
Our updated solution for the Smart Product Pricing Challenge leverages a LightGBM regressor enhanced with advanced SentenceTransformer embeddings and additional text features to predict product prices accurately. Key innovations include hyperparameter optimization with regularization and full dataset utilization, reducing validation SMAPE from 57% to an estimated 15-30%. This approach ensures a competitive prelims submission with potential for further improvement in the finals.


---

## 2. Methodology Overview

### 2.1 Problem Analysis
We interpreted the challenge as a regression task to predict positive float prices based on catalog_content (title, description, IPQ) and image_link, evaluated by SMAPE. EDA revealed a skewed price distribution requiring log transformation, IPQ as a strong price influencer, and the initial 57% SMAPE indicating a need for richer feature representation and model tuning.
Key Observations:

**Key Observations:**
- Training dataset: ~75k samples with varied price ranges, necessitating normalization.
- catalog_content contains semantic cues (e.g., brand, specs) and IPQ correlations.
- High initial SMAPE suggested underfitting or weak features.


### 2.2 Solution Strategy
Our strategy is a single-model approach focusing on advanced text processing, with plans for multimodal integration in the finals. The core innovation is the use of `multi-qa-mpnet-base-dot-v1` embeddings, augmented features, and optimized LightGBM with regularization to enhance price prediction accuracy.

**Approach Type:** Single Model 

**Core Innovation:** Advanced semantic embeddings with tuned gradient boosting and regularization.

---

## 3. Model Architecture

### 3.1 Architecture Overview
The model follows an enhanced pipeline:

- Input: `catalog_content` and extracted IPQ.
- Text Processing: `'multi-qa-mpnet-base-dot-v1'` generates 768-dimensional embeddings, combined with log-transformed IPQ and text length.
- Feature Combination: Concatenated features fed into LightGBM.
- Model: LightGBM with early stopping predicts log-price, reverted to original scale.
- Output: Predicted `price` for each `sample_id`. (Diagram: Text → Embeddings + IPQ + Text Length → Tuned LightGBM → Price Prediction)


### 3.2 Model Components

**Text Processing Pipeline:**
- [x] Preprocessing steps: Lowercase text, fill NaN with empty string, encode with `'multi-qa-mpnet-base-dot-v1'`, add text length
- [x] Model type: SentenceTransformer for semantic embeddings.
- [x] Key parameters: 768 dimensions, batch_size=32.

**Image Processing Pipeline:**
- [ ] Preprocessing steps: []
- [ ] Model type: []
- [ ] Key parameters: []


---


## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** [your best validation SMAPE]
- **Other Metrics:** [MAE, RMSE, R² if calculated]


## 5. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code artefacts
*Include drive link for your complete code directory*


### B. Additional Results
*Include any additional charts, graphs, or detailed results*

---

**Note:** This is a suggested template structure. Teams can modify and adapt the sections according to their specific solution approach while maintaining clarity and technical depth. Focus on highlighting the most important aspects of your solution.