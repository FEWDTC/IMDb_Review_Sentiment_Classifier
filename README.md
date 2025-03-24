# IMDb_Review_Sentiment_Classifier
This project is an end-to-end NLP sentiment analysis pipeline for classifying IMDb movie reviews as **positive** or **negative**. It includes text preprocessing, TF-IDF feature extraction, and evaluation of multiple models including deep learning (LSTM), classical ML (Logistic Regression, Random Forest, XGBoost), and a fully connected Neural Network.


---
## Objective
The goal is to build an effective sentiment analysis model that can **classify IMDb movie reviews** as either **positive (1)** or **negative (0)**. A variety of models are trained and evaluated to understand the trade-offs between traditional and neural network-based approaches.

---
## Dataset

Used the built-in **IMDb movie review dataset** from `tensorflow.keras.datasets`. It contains:

- 25,000 training samples  
- 25,000 test samples  
- Binary sentiment labels:
  - `0`: Negative  
  - `1`: Positive

Only the **10,000 most frequent words** are retained, and sequences are padded/truncated to a fixed length of **256** tokens.

---
## Tools & Technologies

- Python, NumPy, Pandas
- **Scikit-learn** (TF-IDF, Logistic Regression, Random Forest, XGBoost)
- **TensorFlow / Keras** (LSTM, Neural Network)
- **Matplotlib / Seaborn** (Visualization)

---


## Models Implemented

### Traditional Machine Learning (TF-IDF):
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine *(excluded due to long runtime)*

### Deep Learning:
- Basic Neural Network with TF-IDF
- LSTM with Embedding Layer

---
## Model Performance Comparison

| Model                          | Vectorization | Test Accuracy |
|-------------------------------|----------------|---------------|
| **Logistic Regression**        | TF-IDF         | **0.8800**   |
| **Random Forest**              | TF-IDF         | 0.8400       |
| **XGBoost**                    | TF-IDF         | 0.8600       |
| **Basic Neural Network (MLP)** | TF-IDF         | 0.8546       |
| **LSTM (Deep Learning)**       | Embedding Layer | 0.6935      |

**Note:** SVM was excluded due to excessive training time on large TF-IDF matrices.

---

## Visualizations

- Accuracy & Loss over Epochs (for LSTM & NN)
- Confusion Matrices for each model
- Classification Reports (Precision, Recall, F1-Score)

**[View Visualizations](https://github.com/FEWDTC/IMDb_Review_Sentiment_Classifier/blob/main/Notebooks/NLP_based_Sentiment_Analysis.ipynb)**
---

## Key Takeaways

- TF-IDF with **Logistic Regression** outperformed other models with **88% accuracy**
- Deep learning models like **LSTM** struggled slightly due to lack of fine-tuning or pre-trained embeddings
- **XGBoost** and **Basic NN** also achieved strong results above **85% accuracy**


---

## Next Steps

- Add **SVM** with reduced data size or optimized kernel
- Incorporate **pre-trained word embeddings** (GloVe or Word2Vec)
- Deploy best model with **Flask / Streamlit** for live prediction

---

## Author

**Junghyun Cheon (FEWDTC)**  
UC Berkeley, Data Science  
Aspiring Data Analyst / AI Engineer  
[GitHub Profile](https://github.com/FEWDTC)

---

