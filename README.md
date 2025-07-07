# Indonesian Tweet Classification: IndoBERT vs. Generative LLM

[![Python](https://img.shields.io/badge/Python-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers & PEFT](https://img.shields.io/badge/Transformers%20%26%20PEFT-Hugging%20Face-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/docs/transformers/index)

## 📝 Introduction

This project undertakes a comparative study in Natural Language Processing (NLP), focusing on multi-class text classification of Indonesian-language tweets. It implements and evaluates two distinct approaches: fine-tuning a state-of-the-art IndoBERT model and training a generative Large Language Model (LLM) for the same classification task. The primary goal is to analyze the performance trade-offs between these two architectures for categorizing tweets into 8 distinct topics.

Key Technologies: `Python`, `Hugging Face Transformers`, `PyTorch`, `PEFT`.

## 🚀 Project Status

This project is **currently in development**.

-   ✅ The **IndoBERT** model pipeline (preprocessing, fine-tuning, and evaluation) is complete and functional.
-   🚧 The **Generative LLM** component is under active development and experimentation. The goal is to create a comparative analysis against the IndoBERT baseline.

## ✨ Features

-   **Multi-Class Classification:** Categorizes Indonesian tweets into 8 topics (e.g., Economy, Politics, Social Culture).
-   **Advanced Preprocessing:** Implements a comprehensive text preprocessing pipeline tailored for Indonesian, including slang word normalization, noise removal, and stopword removal.
-   **IndoBERT Fine-Tuning:** Utilizes the `indobenchmark/indobert-large-p2` model for high-accuracy classification.
-   **Generative LLM Experimentation:** Explores the use of generative models like `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` for classification tasks.
-   **Performance Analysis:** Provides in-depth model evaluation using metrics like accuracy and confusion matrices.

## 🏗️ Architecture and Workflow

The project follows a structured workflow, from data handling to model evaluation, designed to ensure reproducibility and clear comparison.

1.  **Data Ingestion & Preprocessing (`notebooks/01-data-preprocessing.ipynb`):** Raw tweet data is loaded and undergoes a rigorous cleaning process. This involves normalizing slang, removing irrelevant characters and stopwords, and preparing the text for the models.
2.  **IndoBERT Fine-Tuning (`notebooks/02-model-training-indobert.ipynb`):** The preprocessed dataset is used to fine-tune the `indobenchmark/indobert-large-p2` model. The trained model and its performance metrics are saved for analysis.
3.  **Generative LLM Training (`notebooks/03-llm-training-generative.ipynb`):** The same dataset is formatted and used to train the `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` model, enabling a direct comparison of its classification capabilities.
4.  **Evaluation:** Both models are evaluated on a held-out test set to compare their effectiveness in the classification task.

## 📁 Project Structure

Here is an overview of the project's directory structure. Note that `dataset/`, `logs/`, and `models/` are included in `.gitignore` and will not be present in the repository.

```
.
├── .gitignore
├── LICENSE
├── notebooks/
│   ├── 01-data-preprocessing.ipynb
│   ├── 02-model-training-indobert.ipynb
│   └── 03-llm-training-generative.ipynb
├── dataset/      # (Not in repo)
├── legacy/
│   └── multi-class-nlp.ipynb
├── logs/         # (Not in repo)
├── models/       # (Not in repo)
├── pdm.lock
├── pyproject.toml
└── README.md
```

-   `notebooks/`: Contains the core Jupyter notebooks detailing each step of the workflow.
-   `dataset/`: Intended for storing raw and processed datasets (excluded via `.gitignore`).
-   `legacy/`: Contains older code versions or exploratory notebooks.
-   `logs/`: Stores training logs and outputs (excluded via `.gitignore`).
-   `models/`: Saves trained model artifacts and checkpoints (excluded via `.gitignore`).
-   `pyproject.toml`: Defines project dependencies and metadata for the PDM package manager.

## 🚀 Getting Started

> **Note:** This project is under active development. The following instructions are intended for developers and contributors who wish to work with the current codebase. It is not yet recommended for general use.

### Prerequisites

-   Python 3.12
-   [PDM](https://pdm-project.org/) package manager

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/indonesian-tweet-classification-indobert.git
    cd indonesian-tweet-classification-indobert
    ```

2.  **Install the dependencies using PDM:**
    ```sh
    pdm install
    ```
    This command creates a virtual environment and installs all required packages specified in `pyproject.toml`.

## 💻 Usage

The primary workflow is designed to be executed through the Jupyter notebooks located in the `notebooks/` directory.

To run the project, start a Jupyter server and execute the notebooks in the following order:

1.  `01-data-preprocessing.ipynb`
2.  `02-model-training-indobert.ipynb`
3.  `03-llm-training-generative.ipynb`

## 📦 Note on Data, Models, and Logs

The `dataset/`, `models/`, and `logs/` directories are **not included** in this repository, as specified in the `.gitignore` file. This is standard practice for large files that are not suitable for version control.

-   **Data:** The current data processing pipeline is tailored to a specific, private dataset. As the project is still in development, it has not yet been tested or generalized for use with other datasets.
-   **Models & Logs:** The training scripts will generate the `models/` and `logs/` directories when they are run. You may need to create these directories manually before running the notebooks.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or improvements.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.