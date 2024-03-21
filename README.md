# extremist-classifier
This repository contains the code and resources for a BERT-based classifier for Twitter/X users. This classifier allows to segment users as extremists or moderates.

## Project Overview
The model is trained on a dataset related to political leaning and utilizes the RoBERTa model for sequence classification.

## Scripts
The repository includes the following Python scripts:

- `Report.py`: Contains the `Report` class for generating and saving classification reports.
- `main.py`: The main script that orchestrates the training and evaluation process.
- `Trainer.py`: Defines the `Trainer` class for training the model.
- `get_results.py`: Script for retrieving and saving the classification results.
- `BertClassifier.py`: Defines the `BertClassifier` class, a custom PyTorch model for sequence classification.
- `Dataset.py`: Defines the `CustomDataset` class for loading and preprocessing the dataset.
- `logger_config.py`: Configuration file for setting up the logger.
- `Predictor.py`: Defines the `Predictor` class for making predictions using the trained model.

## Dataset
The datasets used for training and evaluation is not included in this repository as they are too large to contain. Moreover, they include personal information; social media data cannot be publicly posted unless given the ability to hydrate.  Please ensure that you have the necessary dataset files before running the scripts.

## Requirements
To run the scripts, make sure you have the following dependencies installed:
1. Clone the repository: `git clone <repository_url>`
2. Install the required dependencies.
3. Prepare your dataset and ensure it is accessible to the scripts.
4. Run the `main.py` script to start the training and evaluation process.

## Results
The classification results, including precision, recall, F1-score, and accuracy, will be saved in the specified output directory.

## License
This project is licensed under the [MIT License](LICENSE).
