import pandas as pd
import numpy as np

"""
preprocess.py

Purpose:
Handles all raw data cleaning and normalization before feature engineering.

What belongs here:
- Standardizing column names
- Parsing dates into datetime objects
- Cleaning team names so they are consistent across datasets
- Removing or flagging invalid / incomplete rows
- Ensuring data types are correct (ints, floats, strings)

What should NOT go here:
- Feature creation (rolling stats, Elo, etc.)
- Model training or prediction logic

This file should take messy raw data and output a clean,
analysis-ready DataFrame that downstream code can rely on.
"""
