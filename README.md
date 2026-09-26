# Multi-Label Toxic Comment Classification Using PyTorch LSTM

##  Project Overview

This project focuses on **Multi-Label Toxic Comment Classification** using a deep learning model built with **PyTorch LSTM**.

The goal is to automatically identify different types of toxic behavior in online comments.

A single comment can belong to more than one category, making this a **multi-label classification problem**.

### Toxicity Categories

The model predicts six different labels:

* `toxic`
* `severe_toxic`
* `obscene`
* `threat`
* `insult`
* `identity_hate`

---

##  Dataset

The project uses the **Jigsaw Toxic Comment Classification Challenge** dataset.

The dataset contains Wikipedia comments labeled according to different types of toxicity.

The training data contains:

* Comment text
* Six binary toxicity labels

The test set contains **153,164 comments**.

---



##  Preprocessing

The text preprocessing pipeline includes:

* Converting text to lowercase
* Removing URLs
* Removing HTML tags
* Removing non-alphabetic characters
* Removing extra spaces
* Tokenization
* Building a vocabulary
* Converting words into integer IDs
* Padding sequences for batch processing

Unknown words are represented using an `<UNK>` token.

Padding is represented using a `<PAD>` token.

---

##  Handling Class Imbalance

The dataset is highly imbalanced, especially for labels such as:

* `threat`
* `severe_toxic`
* `identity_hate`

To address this issue, the model uses **weighted Binary Cross Entropy with Logits**:

```python
criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)
```

The positive class weights were calculated from the training data.

---


The model was trained using GPU when available.

---

#  LSTM Results

### Accuracy

| Label         | Accuracy |
| ------------- | -------: |
| Toxic         |     0.83 |
| Severe Toxic  |     0.94 |
| Obscene       |     0.85 |
| Threat        |     0.94 |
| Insult        |     0.84 |
| Identity Hate |     0.86 |




---

##  Evaluation

The model was evaluated using:

* Precision
* Recall
* F1-Score
* Accuracy
* Classification Report
* Confusion Matrix

For the evaluation, predictions were converted into binary labels using a threshold of:







##  Project Structure

```text
Toxic-Comment-Classification/
│
├── notebook/
│   └── toxic_comment_lstm.ipynb
│
├── README.md
│
└── requirements.txt
```

---

