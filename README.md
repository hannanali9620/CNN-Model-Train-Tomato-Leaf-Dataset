# Tomato Leaf Disease Detection (CNN)

A deep learning project that identifies **10 tomato leaf conditions** (9 diseases + healthy) from a photo, with a Streamlit dashboard for uploading images and viewing predictions.

Built as a learning project: baseline CNN, regularization experiments, transfer learning (MobileNetV2), then a dashboard.

## Results

Final model: **MobileNetV2 (transfer learning + fine-tuning)**

| Metric | Value |
|---|---|
| Test accuracy | **93.32%** |
| Test loss | 0.196 |
| Test images | 2,724 (never used for model selection) |

### Per-class test results

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bacterial spot | 0.95 | 0.95 | 0.95 | 319 |
| Early blight | 0.88 | **0.61** | 0.72 | 150 |
| Late blight | 0.95 | 0.95 | 0.95 | 286 |
| Leaf mold | 0.93 | 0.94 | 0.94 | 143 |
| Septoria leaf spot | 0.88 | 0.95 | 0.91 | 265 |
| Spider mites | 0.90 | 0.92 | 0.91 | 252 |
| Target spot | 0.77 | 0.95 | 0.85 | 211 |
| Yellow leaf curl virus | 1.00 | 0.96 | 0.98 | 804 |
| Mosaic virus | 0.93 | 0.96 | 0.95 | 56 |
| Healthy | 0.98 | 0.96 | 0.97 | 238 |

**Weakest class: Early blight.** Its leaves were most often mistaken for Target spot (19), Septoria (17) and Late blight (10). These conditions all show spots on the leaf and look similar.

## Experiments (validation set)

| Model | Best val accuracy | Best val loss |
|---|---|---|
| Baseline CNN (3 conv blocks, 4.29M params) | 93.3% | 0.238 |
| CNN + BatchNorm + Dropout + heavy augmentation (112K params) | 87.5% | 0.328 |
| MobileNetV2, frozen base (head only) | 85.2% | 0.439 |
| MobileNetV2, fine-tuned (BatchNorm trainable) | 88.8% | 0.372 |
| **MobileNetV2, fine-tuned (BatchNorm frozen)** | **93.8%** | **0.183** |

What I learned from these runs:
- The baseline overfit: training accuracy reached ~98% while validation loss stopped improving after epoch 5.
- Adding every regularizer at once made the small model **underfit** (train accuracy stuck near 88%).
- Fine-tuning with the BatchNorm layers trainable made the loss jump at the start of fine-tuning. Freezing them fixed it.
- The baseline's best-epoch weights were not saved, so the baseline was not evaluated on the test set.

## Dataset

[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (color images), tomato subset: 10 classes, 18,160 images.

- Stratified split: 70% train / 15% validation / 15% test (12,712 / 2,724 / 2,724)
- Images resized to 128x128
- Class imbalance (373 to 5,357 images per class) handled with class weights

The dataset is not included in this repo.

## Training approach

1. Frozen MobileNetV2 (ImageNet weights) + new classifier head, lr 1e-3
2. Fine-tune the last 30 layers of the base with BatchNorm layers frozen, lr 3e-5
3. Light augmentation (flip, rotation, zoom), EarlyStopping, ReduceLROnPlateau, class weights

## Run the dashboard

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

Use the same TensorFlow version that was used for training, otherwise loading the model may fail.

## Project structure

    app.py                  Streamlit dashboard
    tomato_tl2.keras        Trained model
    class_names.json        Class labels
    requirements.txt
    .streamlit/config.toml  Dashboard theme
    notebooks/              Training notebook

## Limitations

- PlantVillage images are lab-style (clean background, even light). **Accuracy on real field photos can be lower.** This was not tested.
- The model only knows these 10 tomato conditions. For any other photo it will still return one of them.
- This is an estimate, not a diagnosis. Confirm with an agricultural expert before making crop decisions.

## Tech

Python, TensorFlow / Keras, scikit-learn, Streamlit, Google Colab (T4 GPU)
