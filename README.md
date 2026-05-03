# Image Classification — DSAI Lab Week 3 (Bonus Assignment)

**Course:** IIT Madras BS in Data Science & Applications — JAN 2026 Term  
**Competition:** [Image Classification | DS&AI | 26T1 Week 3](https://www.kaggle.com/competitions/image-classification-ds-ai-26-t-1-week-3) on Kaggle  
**Metric:** Weighted F1-Score  
**Result:** Validation Weighted F1 = **0.9340 (93.4%)**

---

## What is this project?

Given a photo, predict which of 6 scene categories it belongs to:

| Class | Example scene |
|-------|---------------|
| `buildings` | Skylines, cityscapes |
| `forest` | Trees, dense vegetation |
| `glacier` | Ice fields, snowy landscapes |
| `mountain` | Rocky peaks, hills |
| `sea` | Ocean, coastal views |
| `street` | Roads, urban streets |

---

## Dataset

> Dataset is available via the [Kaggle competition page](https://www.kaggle.com/competitions/image-classification-ds-ai-26-t-1-week-3/data) (requires competition enrollment).

| Split | Count |
|-------|-------|
| Training images | 14,034 |
| Test images | 3,000 |

Training images are organized into class folders (`train/train/buildings/`, `train/train/forest/`, etc.).  
Test images are named `img000.jpg` → `img2999.jpg` with no labels.

The dataset is slightly imbalanced (`mountain` has the most images, `buildings` the fewest), so class weights are used during training to handle this.

---

## Approach

### Model: MobileNetV2 + Custom Head

A pre-trained **MobileNetV2** (trained on ImageNet) is used as the feature extractor. A custom classification head is added on top:

```
Input (224×224×3)
   → MobileNetV2 backbone (frozen in Phase 1, partly unfrozen in Phase 2)
   → GlobalAveragePooling2D
   → BatchNorm → Dropout(0.4)
   → Dense(512, ReLU) → BatchNorm → Dropout(0.2)
   → Dense(6, Softmax)
```

Total parameters: ~2.9M (only ~660K trainable in Phase 1).

### Two-Phase Training

**Phase 1 — Train the head only (backbone frozen)**
- Epochs: up to 12, early stopping patience 5
- Optimizer: Adam, LR = 3e-4
- Best val accuracy: ~91.4%

**Phase 2 — Fine-tune top 100 backbone layers**
- Epochs: up to 18, early stopping patience 7
- Optimizer: Adam with Cosine Decay Restarts schedule, initial LR = 5e-5
- Best val accuracy: ~93.4%

### Data Augmentation (Training)

Applied on-the-fly during training to reduce overfitting:
- Random horizontal flip
- Random rotation (±15°)
- Random zoom (±15%)
- Random contrast and brightness (±15%)
- Random translation (±10%)

### Test-Time Augmentation (TTA)

At inference time, each test image is passed through the model **9 times** (1 clean + 8 augmented). The predicted probabilities are averaged before taking the final class. This typically improves accuracy by 1–2%.

---

## Results

| Class | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| buildings | 0.94 | 0.92 | 0.93 |
| forest | 0.99 | 0.99 | 0.99 |
| glacier | 0.87 | 0.88 | 0.87 |
| mountain | 0.91 | 0.89 | 0.90 |
| sea | 0.97 | 0.98 | 0.97 |
| street | 0.93 | 0.94 | 0.94 |
| **Weighted avg** | **0.93** | **0.93** | **0.93** |

Validation set size: 2,806 images (20% of training data).

---

## Files

```
Bonus Assignment Week 3/
├── kaggle_submission_notebook.ipynb   # Main notebook (run this on Kaggle)
├── notebook0a1a6c474a.ipynb           # Same notebook with Kaggle run outputs
└── image-classification-ds-ai-26-t-1-week-3/
    ├── sample_submission.csv          # Submission format template (3000 rows)
    └── test/test/                     # Test images (img000.jpg … img2999.jpg)
```

Training images are only available inside the Kaggle competition environment (not stored locally).

---

## How to Run (on Kaggle)

1. Join the competition and go to the **Code** tab → **New Notebook**.
2. Change your team name to your **institute roll number**.
3. In the notebook sidebar, attach the competition dataset as input.
4. Copy the code from `kaggle_submission_notebook.ipynb` and run all cells.
5. The notebook automatically saves `submission.csv` to `/kaggle/working/`.
6. Click **Submit** on Kaggle using that CSV file.

> **Important:** Submissions must be made through Kaggle Notebooks only. Locally trained models are not accepted.

---

## Submission Format

```
id,target
img000,buildings
img001,forest
img002,glacier
...
img2999,street
```

---

## Key Libraries

- `TensorFlow / Keras` — model building and training
- `scikit-learn` — class weights, F1-score, classification report
- `Pandas / NumPy` — data handling
- `Pillow` — image loading for visualization
