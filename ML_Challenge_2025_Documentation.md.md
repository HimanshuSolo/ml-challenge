# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Neural Nexus

**Team Members:** Himanshu(Team Leader), Vaibhav, Jigisha Saigal, Yashika Moni

**Submission Date:** 12-10-2025

---

## 1. Executive Summary
Our updated solution for the Smart Product Pricing Challenge employs a multimodal LightGBM regressor, combining advanced SentenceTransformer embeddings, IPQ, text length, and ResNet-50 image features to predict product prices accurately. Key innovations include hyperparameter optimization with regularization and full dataset utilization, reducing validation SMAPE from 57% to an estimated 10-20%. This approach positions us competitively for the prelims with potential for further enhancement in the finals.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
We interpreted the challenge as a regression task to predict positive float prices based on `catalog_content` (title, description, IPQ) and `image_link`, evaluated by SMAPE. EDA revealed a skewed price distribution requiring log transformation, IPQ as a strong price influencer, and the initial 57% SMAPE indicating a need for richer feature representation and multimodal data integration.

**Key Observations:**
- Training dataset: ~75k samples with varied price ranges, necessitating normalization.
- `catalog_content` contains semantic cues (e.g., brand, specs) and IPQ correlations.
- `image_link` offers visual context (e.g., product quality), underutilized initially.

### 2.2 Solution Strategy
Our strategy is a multimodal single-model approach, leveraging text and image processing, with plans for ensemble methods in the finals. The core innovation is the integration of `'multi-qa-mpnet-base-dot-v1'` embeddings, ResNet-50 image features, and optimized LightGBM with regularization to enhance price prediction accuracy.

**Approach Type:** Multimodal Single Model  
**Core Innovation:** Combined semantic text embeddings, image features, and tuned gradient boosting.

---

## 3. Model Architecture

### 3.1 Architecture Overview
The model follows an enhanced multimodal pipeline:  
- **Input**: `catalog_content` and `image_link`.  
- **Text Processing**: `'multi-qa-mpnet-base-dot-v1'` generates 768-dimensional embeddings, combined with log-transformed IPQ and text length.  
- **Image Processing**: ResNet-50 extracts 1000-dimensional features from downloaded images (224x224, normalized).  
- **Feature Combination**: Concatenated features fed into LightGBM.  
- **Model**: LightGBM with early stopping predicts log-price, reverted to original scale.  
- **Output**: Predicted `price` for each `sample_id`.  
*(Diagram: Text → Embeddings + IPQ + Text Length | Images → ResNet-50 → Combined Features → Tuned LightGBM → Price Prediction)*

### 3.2 Model Components

**Text Processing Pipeline:**
- [x] Preprocessing steps: Lowercase text, fill NaN with empty string, encode with `'multi-qa-mpnet-base-dot-v1'`, add text length.
- [x] Model type: SentenceTransformer for semantic embeddings.
- [x] Key parameters: 768 dimensions, batch_size=32.

**Image Processing Pipeline:**
- [x] Preprocessing steps: Download images, resize to 224x224, normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225], handle errors with zero vectors.
- [x] Model type: ResNet-50 (pretrained).
- [x] Key parameters: 1000-dimensional output, batch processing via fallback `download_images`.

---

## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** 18%

## 5. Conclusion
Our multimodal approach reduced SMAPE from 57% to an estimated 10-20% by integrating advanced text embeddings, image features, and optimized LightGBM, highlighting the power of visual and semantic data. A key lesson is the importance of balancing computational cost with feature richness, especially under time constraints. For the finals, we plan to refine image processing and explore ensemble techniques to target even lower SMAPE.

---

## Appendix

### A. Code Artefacts
[Code Output](https://drive.google.com/file/d/1ZesyEuM7FMuN0KGlEC3yjVzhOKC7-muD/view?usp=sharing) 

### B. Additional Results
*Include any additional charts, graphs, or detailed results*  
- [Example: Plot of CV SMAPE vs. hyperparameters, or multimodal feature impact] *(If time allows, generate with matplotlib and add the link or embed the image)*

---