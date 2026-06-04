# Traffic Demand Prediction - ML Solution

**[Hackathon Link](https://www.hackerearth.com/challenges/competitive/gridlock-hackathon-20/machine-learning/traffic-demand-prediction-12-b86d1caf/)**

## Overview
This project predicts traffic demand at different locations and times using machine learning. The goal is to forecast passenger demand patterns to optimize traffic management and resource allocation.

## Dataset Description

### Files
- `train.csv` - Training dataset with 77,299 records
- `test.csv` - Test dataset for predictions
- `sample_submission.csv` - Submission format template

### Features

| Feature | Type | Description |
|---------|------|-------------|
| Index | int | Unique identification for each datapoint |
| geohash | string | Geographic information encoded as geohash |
| day | int | Day when information was recorded (48 or 49) |
| timestamp | string | Time in HH:MM format |
| RoadType | string | Type of road (Residential, Street, Highway) |
| NumberofLanes | int | Number of lanes at the location (1-5) |
| LargeVehicles | string | Large vehicles permitted/not |
| Landmarks | string | Presence of landmarks nearby (Yes/No) |
| Temperature | float | Temperature at the location (°C) |
| Weather | string | Weather condition (Sunny, Rainy, Snowy, Foggy) |
| **demand** | float | **Target variable** - Traffic demand (0-1 normalized) |

## Solution Approach

### 1. Data Preprocessing
- **Missing Value Handling:**
  - Temperature: Filled with median (16.38°C)
  - RoadType: Filled with 'Unknown' category
  - Weather: Filled with 'Unknown' category
  
### 2. Feature Engineering
- **Temporal Features:**
  - Extracted hour and minute from timestamp
  - Created cylindrical features (hour_sin, hour_cos) to preserve time circularity
  - Quarter-hour feature for granular time buckets
  
- **Geospatial Features:**
  - Decoded geohash to latitude and longitude using pygeohash library
  - Enables geographic pattern recognition
  
- **Categorical Encoding:**
  - Binary encoding: LargeVehicles, Landmarks
  - One-hot encoding: RoadType (3 categories), Weather (4 categories)

### 3. Target Transformation
- Applied log1p transformation to demand
- Addresses highly skewed distribution (range: 0.0000006 to 1.0)
- Improves model convergence and prediction accuracy

### 4. Model Architecture

**Algorithm:** LightGBM Regressor

**Configuration:**
```python
{
  'n_estimators': 1000,
  'learning_rate': 0.05,
  'num_leaves': 31,
  'random_state': 42,
  'feature_fraction': 0.8,
  'bagging_fraction': 0.8,
  'reg_alpha': 0.1,      # L1 regularization
  'reg_lambda': 0.1      # L2 regularization
}
```

**Training Strategy:**
- Train-test split: 80/20
- Early stopping: Monitor test loss, stop if no improvement for 100 rounds
- Prevents overfitting while ensuring convergence

### 5. Evaluation Metrics
The model is evaluated on:
- **MAE** (Mean Absolute Error) - Captures average prediction error
- **RMSE** (Root Mean Squared Error) - Penalizes larger errors
- **R² Score** - Measures proportion of variance explained

### 6. Feature Importance
Top contributing features:
1. Latitude/Longitude (geographic patterns)
2. RoadType (Residential heavily influences demand)
3. NumberofLanes (capacity indicator)
4. Hour features (temporal patterns)
5. LargeVehicles (regulatory constraints)

## Project Structure
```
FlipKartGrid/
├── README.md              # Project documentation
├── .gitignore
├── uv.lock
├── .python-version
├── pyproject.toml        # Project configuration
├── main.py               # Entry point
├── scripts               # Streamlit Web
├── src/
│   └── exp.ipynb         # Complete ML pipeline notebook
└── data/
    ├── train.csv         # Training data
    ├── test.csv          # Test data
    ├── sample_submission.csv
    └── submission.csv    # Final predictions
```

## Setup & Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 0.24.0
- lightgbm >= 3.3.0
- pygeohash >= 1.2.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Usage

### Run the Notebook
```bash
jupyter notebook src/exp.ipynb
```

The notebook executes the complete pipeline:
1. Load and explore data
2. Clean and preprocess features
3. Engineer new features
4. Train the model with early stopping
5. Evaluate on test set
6. Generate predictions and submission file

### Output
- `data/submission.csv` - Predictions with Index and demand columns

## Key Insights

### Demand Patterns
- **Road Type Impact:** Street roads show 29% higher correlation with demand than residential areas
- **Time of Day:** Hour_sin feature shows significant temporal patterns
- **Geographic Variation:** Latitude-based clustering indicates regional demand differences
- **Weather Impact:** Minimal direct correlation; combined effect with other features more important

### Model Performance
- Captures non-linear relationships through tree-based learning
- Regularization prevents overfitting (reg_alpha=0.1, reg_lambda=0.1)
- Early stopping balances model complexity with generalization

## Future Improvements
1. **Advanced Feature Engineering:**
   - Interaction features (hour × RoadType, weather × temperature)
   - Geographical clustering using K-means
   - Cyclical features for seasonal patterns
   
2. **Model Enhancements:**
   - Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
   - Ensemble methods (stacking, boosting)
   - Cross-validation for robust evaluation
   
3. **Outlier Handling:**
   - Detect and handle extreme demand values
   - IQR-based or statistical methods
   
4. **Domain-Specific Features:**
   - Rush hour detection
   - Holiday/weekend indicators
   - Special events marking

## References
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Geohash Explanation](https://en.wikipedia.org/wiki/Geohash)
- [Feature Engineering Techniques](https://scikit-learn.org/stable/)
