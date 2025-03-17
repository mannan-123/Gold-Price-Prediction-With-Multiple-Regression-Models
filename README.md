# 📈 Gold Price Prediction Using Multiple Regression Models

## **Introduction**

Gold is one of the most valuable commodities in global financial markets, and its price is influenced by multiple factors such as inflation, currency exchange rates, and geopolitical events. Predicting gold prices accurately can help investors and policymakers make informed decisions.

In this project, we use **multiple regression models** to analyze and predict gold prices based on historical data. We implement and compare the following machine learning algorithms:

- **Decision Tree & Random Forest Regression**
- **Gradient Boosting (GBR, XGBoost, LightGBM)**

The goal is to identify the most accurate model for gold price prediction by evaluating key performance metrics such as **Mean Squared Error (MSE), and R² Score**.

## **Dataset**

The dataset used in this project is sourced from Kaggle: [Gold Price Data](https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data). It contains historical records of gold prices along with other financial indicators.

## **Project Structure**

```
Gold-Price-Prediction/
│-- Gold_Price_Prediction.ipynb    # Jupyter Notebook
│-- README.md                      # Project documentation
```

## **Installation & Setup**

1. Clone the repository:
   ```bash
   https://github.com/mannan-123/Gold-Price-Prediction-With-Multiple-Regression-Models.git
   cd Gold-Price-Prediction-With-Multiple-Regression-Models
   ```
2. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm kagglehub
   ```
3. Download the dataset using Kaggle API:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("altruistdelhite04/gold-price-data")
   ```

## **Exploratory Data Analysis (EDA)**

The dataset is examined through various statistical and visualization techniques, including:

- **Checking missing values and data types**
- **Correlation heatmaps**
- **Time series visualization of gold prices**
- **Histograms and boxplots**

### Example Visualization:

```python
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```

## **Feature Engineering & Data Preprocessing**

- Extracting date-related features (Year, Month, Day, DayOfWeek)
- Handling missing values (if any)
- Scaling numerical features for models that require it

## **Model Training & Evaluation**

We train and evaluate multiple regression models:

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
dt_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)
```

## **Model Comparison & Results**

```python
def evaluate_model(X_test, y_test):
    models = {"Decision Tree": dt_model, "Random Forest": rf_model, "Gradient Boosting": gbr_model, "XGBoost": xgb_model, "LightGBM": lgbm_model}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: MSE = {mse:.2f}, R² = {r2:.2f}")

evaluate_model(X_test, y_test)
```

| Model             | MSE   | R²   |
| ----------------- | ----- | ---- |
| Decision Tree     | 3.48  | 0.99 |
| Random Forest     | 3.28  | 0.99 |
| Gradient Boosting | 17.50 | 0.97 |
| XGBoost           | 5.16  | 0.99 |
| LightGBM          | 6.40  | 0.99 |
