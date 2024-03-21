# Political ideology Classification
This repository contains the code and resources for an academic project that aims to classify Twitter users based on their political ideology, distinguishing between moderates and extremists among left-leaning and right-leaning users.

## Project Overview
The project utilizes two main datasets: (not available on this repository for privacy concerns)
1. Tweets of users
2. Ideal point estimates (ideological scores) from the Barbera (2015).
The goal is to find the ideal point estimates on a 2-dimensional scale that best distinguishes between moderates and extremists within each political leaning (left and right). The classification is based on signals produced by the tweets of each user, which are used to identify differences between moderates and extremists.

## Methodology
1. Data Preprocessing:
   - The tweets and ideal point estimates datasets are preprocessed and combined to create a unified dataset for analysis.

2. Feature Extraction:
   - Relevant features are extracted from the tweets of each user to capture signals that may indicate their political ideology and extremity.

3. Model Training:
   - The project utilizes the RoBERTa model for sequence classification to train a classifier on the preprocessed dataset.
   - The model is trained to distinguish between moderates and extremists within each political leaning (left and right).

4. Evaluation:
   - The trained model is evaluated on a test dataset to assess its performance in classifying users based on their political ideology and extremity.
   - Metrics such as precision, recall, F1-score, and accuracy are used to measure the model's performance.

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

## Results
The classification results, including precision, recall, F1-score, and accuracy, will be saved in the results folder. This folder contains the results for three ideological thresholds among left-leaning and right-leaning users: 0.3, 0.5, and a 0.9 threshold. Such thresholds identify the proportion of users on either side of the extremist/moderate divide.

## License
This project is licensed under the [MIT License](LICENSE).
