import pandas as pd
import numpy as np
from collections import defaultdict

"""
features.py

Purpose:
Creates model-ready features from cleaned game data.

What belongs here:
- Rolling team statistics (last N games)
- Season-to-date averages
- Point margin differentials
- Home vs away comparisons
- Elo ratings or other team strength metrics

Key rule:
All features must be computed using ONLY information available
before the game date (no data leakage).

What should NOT go here:
- Model training
- File I/O (reading/writing CSVs)
- Submission formatting

This file converts cleaned historical data into numerical
features that a model can learn from.
"""
