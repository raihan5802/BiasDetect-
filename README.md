# BiasDetect - Gendered Language Bias Detection and Text Classification

BiasDetect is a project designed for text classification and sentiment analysis, focusing on detecting biases and analyzing gendered language in passages. This project uses various data preprocessing, feature engineering, and a neural network-based model to classify text into gender-related categories.

## Project Structure

### 1. Data Loading and Inspection
- **Dataset Overview**: Loads and inspects the dataset to identify structure, data types, and missing values.
- **Class Distribution**: Analyzes class counts to check for potential class imbalance.

### 2. Text Preprocessing
- **Text Cleaning**: Converts text to lowercase and removes non-alphanumeric characters, standardizing entries.
- **Stopword Removal**: Optionally removes common stopwords to reduce noise.

### 3. Feature Engineering
- **Keyword and Phrase Detection**: Creates binary features based on specific keywords and phrases tied to each class (e.g., “man,” “woman,” “non-binary”).
- **Keyword Proximity**: Captures contextual relationships by checking if keywords appear within five words of each other.
- **Sentiment Analysis**: Uses TextBlob to calculate text polarity, capturing sentiment for each passage.

### 4. Text Vectorization
- **TF-IDF Vectorization**: Transforms cleaned text into dense numerical representations, emphasizing term importance.
- **Dimensionality Reduction (PCA)**: Applies Principal Component Analysis to reduce feature dimensionality, improving computational efficiency.

### 5. Model Definition
- **Minimal Neural Network Architecture**: Defines a compact neural network for classifying text into four classes, optimizing for interpretability and avoiding overfitting.

### 6. Model Training
- **Training and Validation Loops**: Implements training with Cross Entropy Loss and the Adam optimizer, tracking validation performance with accuracy and F1-score metrics.
- **Metrics Calculation**: Saves the best model based on validation F1-score.

### 7. Model Evaluation and Metrics
- **F1NOP Calculation**: Computes a custom F1NOP score that balances F1 performance with model size, providing insights into model efficiency.

### 8. Model and Prediction Saving
- **Model Saving**: Saves the best-performing model and final predictions to designated files for reproducibility.

## Requirements
- **Libraries**: pandas, re, sklearn, torch, TextBlob, joblib, thop, numpy

## Usage
1. Load and preprocess the dataset.
2. Run feature engineering and text vectorization.
3. Train the model using the training loop provided.
4. Evaluate model performance and save the final predictions.


