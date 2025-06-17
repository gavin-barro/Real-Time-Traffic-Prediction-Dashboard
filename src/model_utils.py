# Train, evaluate, save models

import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """
    Returns MAE, RMSE, and R²
    
    Parameters:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target values

    Returns:
        tuple[float, float, float]: MAE, RMSE, R²
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def show_plots(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
        Displays a scatter plot comparing true vs. predicted values
    
        Parameters:
            y_test (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
        
        Returns:
            None
    """
    
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Volume")
    plt.ylabel("Predicted Volume")
    plt.title("Actual vs. Predicted Traffic Volume")
    plt.tight_layout()
    plt.show()

def save_model(pipeline: Pipeline, filename: str = "models/traffic_model.pkl") -> None:
    """
    Saves the trained model pipeline to a file.
    
    Parameters:
        pipeline (Pipeline): Trained pipeline object
        filename (str): File path to save the model
    """
    
    joblib.dump(pipeline, filename)

def load_model(filename: str = "models/traffic_model.pkl") -> Pipeline:
    """
    Loads a trained model pipeline from a file.
    
    Parameters:
        filename (str): Path to saved model file

    Returns:
        Pipeline: Loaded pipeline object
    """
    
    return joblib.load(filename)