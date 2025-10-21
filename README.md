<div align="center">

# Multimodal Price Prediction Using Text & Image Fusion with LGBMRegressor

[Debargha Mitra Roy][def7] &emsp; [Arpan Pramanick][def8] &emsp; [Rounak Koner][def9] &emsp; [Unnati Mishra][def10]

National Institute of Technology, Durgapur

[![Submission][def11]][def12]

</div>

> Please refer to the [Problem Statement][def6] to gain a clear and comprehensive understanding of the problem. This problem was featured in the [Amazon ML Challenge 2025][def14].

## Executive Summary

Our solution involves the use of a multimodal machine learning model, which combines the textual and visual data about products in order to precisely estimate the best prices. With a SentenceTransformer-based text embedding, ResNet50 visual representations, and a LightGBM regression model, we are able to sum up the semantic and aesthetic indicators that affect price. Such a hybrid approach provides more interpretability, scalability and predictive accuracy across different product categories.

## Methodology Overview

### Problem Analysis

The problem aimed to predict optimal product prices using multimodal data - combining textual catalog descriptions and product images. The dataset contained missing text fields, which were filled with empty strings, and all text was standardized to lowercase.
Exploratory analysis revealed that textual content encoded product category, brand, and quality cues, while visual data offered complementary insights like color, texture, and design. Together, these modalities provide a holistic representation of the product, making a multimodal strategy ideal.

**Key Observations:**

The fusion of textual and visual features significantly improved price prediction accuracy compared to single-modality models. Text embeddings captured semantic cues like brand and quality, while image features added visual context, resulting in a balanced and robust multimodal pricing model.

### Solution Strategy

We are proposing to use a multimodal machine learning model, which will unite the textual and the visual data concerning products with the aim of accurately approximating the most appropriate prices. We can combine semantic and aesthetic indicators which influence price with a SentenceTransformer-based text embedding, ResNet50 visual representations, and a LightGBM regression model. A hybrid strategy offers higher interpretation, scale and predictive validity between various product lines.

**Approach Type:** Hybrid Multimodal Model

**Core Innovation:** Fusion of SentenceTransformer text embeddings and ResNet50 visual features, trained with a LightGBM Regressor for accurate price prediction.

**Workflow Summary:**

1. Text preprocessing and embedding generation using SentenceTransformer.
2. Image feature extraction using pre-trained ResNet50.
3. Fusion of multimodal embeddings.
4. Training `LightGBM` on combined features.
5. Model evaluation using MAE, RMSE, and R² metrics.

## Model Architecture

### Architecture Overview

The model comprises three major components:

- **Text Processing Pipeline:** Encodes product descriptions into semantic vectors.
- **Image Processing Pipeline:** Extracts high-level visual embeddings.
- **Fusion + Regression Layer:** Concatenates both embeddings and passes them to LightGBM for final price prediction.

[![Model Architecture][def1]][def1]

### Model Components

**Text Processing Pipeline:**

- Preprocessing steps: Lowercasing, missing value handling, cleaning.
- Model type: `SentenceTransformer(all-MiniLM-L6-v2)` ($384$-dim embeddings).
- Key parameters: Dense vector per sentence capturing context and brand-specific meaning.

**Image Processing Pipeline:**

- Preprocessing steps: Resize $(224×224)$, normalization via `preprocess_input()`.
- Model type: Pre-trained ResNet50 (without top layers).
- Key parameters: $2048$-dim feature vector from global average pooling.

**Fusion + Regression:**
Features concatenated using scipy.sparse.hstack().
Model: LightGBMRegressor (n_estimators=1000, learning_rate=0.05).
Objective: Regression task on price variable.

[![Model Architecture][def13]][def13]

## Model Performance

### Validation Results

- **SMAPE Score:** $22.85%$
- **Other Metrics:** Calculated $R^{2}, MAE, RMSE$.
  - $R^{2}: 0.58$
  - $MAE: 0.58$
  - $RMSE: 0.58$

## Conclusion

The proposed Smart Product Pricing System is an excellent blend between language and vision intelligence to make the right predictions regarding prices. The model is expected to have excellent generalization and interpretability with the combination of SentenceTransformer-based textual understanding, ResNet50-based visual representation, and LightGBM regression.

This is a scalable and adaptable hybrid methodology that offers a strong structure of intelligent pricing automation across various e-commerce platforms.

## Appendix

|         Code Artefacts          |            Links             |
| :-----------------------------: | :--------------------------: |
|  Kaggle Dataset Download Link   | [![Kaggle Link][def2]][def3] |
| Uploaded Code Google Drive Link | [![Drive Link][def4]][def5]  |

[def1]: ./model_architecture/model_architecture.png
[def2]: https://img.shields.io/badge/Kaggle-%23181717?style=flat&logo=kaggle&logoColor=%2320BEFF
[def3]: https://www.kaggle.com/datasets/debarghamitraroy/amazon-catalog-price-dataset
[def4]: https://img.shields.io/badge/Google%20Drive-%23181717?style=flat&logo=googledrive&logoColor=%234285F4
[def5]: https://drive.google.com/drive/folders/1pXO56Ne4bZ_gqFoJkhchE8nytyLnAmrG?usp=drive_link
[def6]: ./Problem%20Statement.pdf
[def7]: https://www.linkedin.com/in/debarghamitraroy/
[def8]: https://www.linkedin.com/in/arpan-pramanick-7346b6228/
[def9]: https://www.linkedin.com/in/rounak-koner-6279b922a/
[def10]: https://www.linkedin.com/in/unnati-mishra-b1a848233/
[def11]: https://img.shields.io/badge/Submission-Multimodal%20Price%20Prediction%20Using%20Text%20and%20Image%20Fusion%20with%20LGBMRegressor-red?style=flat&logo=googledocs&logoColor=red
[def12]: ./Multimodal%20Price%20Prediction%20Using%20Text%20and%20Image%20Fusion%20with%20LGBMRegressor.pdf
[def13]: ./model_architecture/model_architecture.jpg
[def14]: https://unstop.com/hackathons/amazon-ml-challenge-2025-amazon-1560375
