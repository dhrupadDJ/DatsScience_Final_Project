# Regression Models Project

## Description
This project involves the implementation of various regression models to predict outcomes based on a given dataset. It includes Linear Regression, Decision Tree, and Random Forest models. The project is structured to be modular, with separate components for data loading, preprocessing, model training, evaluation, and visualization.

## Project Structure
- `data_preprocessing/`: Contains modules for data loading and preprocessing.
  - `load_data.py`: Defines function to load data from CSV.
  - `preprocess_data.py`: Includes functions to prepare and split the data.
- `models/`: Contains different model training and evaluation scripts.
  - `linear_regression.py`: Module for training and evaluating Linear Regression.
  - `decision_tree.py`: Module for training Decision Trees and plotting.
  - `random_forest.py`: Module for training and evaluating Random Forest.
- `utils/`: Utility scripts for model persistence.
  - `save_model.py`: Functions to save and load models.
- `main.py`: The main script that orchestrates the loading, processing, training, and evaluation of models.

## Requirements
- Python 3.8 or above
- Libraries: pandas, scikit-learn, matplotlib (for plotting decision trees)
- To install all necessary libraries, run:
  ```bash
  pip install -r requirements.txt
---------------------------------------------------------------------------------------------------------
Run the Main Script:
Navigate to the project directory and run:

python main.py

Output:
The script will display the Mean Absolute Error (MAE) for training and testing sets of each model.
Decision Tree model will also generate a plot of the tree.
----------------------------------------------------------------------------------------------------------
Data
The project uses a dataset stored in final.csv, which should be placed in the root directory of the project or modified in main.py to point to the correct location.

Saving and Loading Models
The Random Forest model is saved to disk after training. It can be loaded and used for predictions without retraining.
Sample Prediction
After loading the model, a sample prediction is carried out using predefined sample data.

Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.