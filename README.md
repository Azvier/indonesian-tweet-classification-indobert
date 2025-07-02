# Multi-Class Indonesian Tweet Classification with IndoBERT

This project focuses on Natural Language Processing (NLP), specifically multi-class text classification. It uses a state-of-the-art pre-trained transformer model, IndoBERT, to categorize Indonesian-language tweets into 8 distinct topics (e.g., Economy, Politics, Social Culture).

## Technical Overview

-   **Transformer Model:** Developed a text classification model utilizing the IndoBERT pre-trained model from the Hugging Face ecosystem, which is specifically trained on a large Indonesian corpus.
-   **Text Preprocessing Pipeline:** Implemented a comprehensive text preprocessing pipeline tailored for Indonesian text. This included slang word normalization, noise removal (using regex), stopword removal, and specialized tokenization for the IndoBERT architecture.
-   **Model Fine-Tuning:** Performed a fine-tuning process on the IndoBERT model using the preprocessed dataset, successfully achieving a classification accuracy of 78%.
-   **Performance Analysis:** Analyzed the model's performance in-depth using a confusion matrix to understand its effectiveness and identify strengths and weaknesses in classifying each specific topic.

## Technologies & Libraries Used

-   Python
-   Transformers (Hugging Face)
-   PyTorch / TensorFlow
-   Pandas
-   NumPy
-   Scikit-learn (for metrics)

## Project Workflow

Raw Tweet Data → Text Preprocessing (Cleaning, Normalization) → IndoBERT Tokenization → Model Fine-Tuning → Performance Evaluation (Accuracy, Confusion Matrix)

---

***Disclaimer:** Please note that the code currently in this repository is the raw version from an earlier stage of my learning journey. A more well-documented, refactored, and structured version is currently in development.*
