import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib

"""
train.py

Purpose:
Trains a point spread prediction model using historical game data.

What this file does:
- Loads cleaned and feature-engineered data
- Splits data into training and validation sets (time-based)
- Trains one or more regression models
- Evaluates model performance using MAE or similar metrics
- Saves the best-performing model to disk

Outputs:
- A trained model file saved in the models/ directory
- (Optionally) metadata like feature column order

What should NOT go here:
- Submission CSV creation
- Hardcoded future games

This script is run whenever you want to retrain or improve the model.
"""
