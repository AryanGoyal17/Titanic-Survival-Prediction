# Titanic Survival Prediction: A Machine Learning Pipeline 🚢

This repository contains a machine learning project that predicts passenger survival on the Titanic. It demonstrates a complete data science workflow, from raw data exploration and cleaning to the training, tuning, and evaluation of multiple classification algorithms.

## 📂 Project Structure

The project is organized sequentially to reflect the data science pipeline:

    ├── data/
    │   ├── Titanic_dataset.csv       # The raw, unprocessed data
    │   └── Titanic_processed.csv     # The cleaned data, ready for modeling
    └── notebooks/
        ├── 01_titanic_eda.ipynb            # Exploratory Data Analysis & Preprocessing
        ├── 02_logistic_regression.ipynb    # Baseline linear modeling & scaling
        └── 03_random_forest.ipynb          # Non-linear modeling & hyperparameter tuning


## 🧠 Workflow & Methodology

### Phase 1: Exploratory Data Analysis & Preprocessing (`01_titanic_eda.ipynb`)
* **Missing Data Handling:** Identified missing values using Seaborn heatmaps. Dropped the highly incomplete `Cabin` feature and mathematically imputed `Age` based on the median ages across different passenger classes (`Pclass`) to preserve feature correlation.
* **Feature Engineering:** Converted categorical text variables (`Sex`, `Embarked`) into numerical format using pandas `get_dummies`, explicitly dropping the first column to avoid the **Dummy Variable Trap** (multicollinearity).
* **Data Visualization:** Analyzed survival rates across gender, socio-economic class, and age distributions to establish early hypotheses.

### Phase 2: Logistic Regression (`02_logistic_regression.ipynb`)
* **Data Scaling:** Applied `StandardScaler` to ensure features like `Fare` and `Pclass` were evaluated on the same mathematical scale, optimizing the algorithm's convergence efficiency.
* **Evaluation:** Achieved a baseline accuracy of **~79.8%**.
* **Analysis:** The classification report revealed class imbalance effects; the model was highly proficient at identifying non-survivors (85% Precision) but slightly less confident with survivors (72% Precision).

### Phase 3: Random Forest Classifier (`03_random_forest.ipynb`)
* **Algorithm Selection:** Deployed a non-linear Random Forest algorithm to capture complex, underlying relationships that straight-line models might miss.
* **Hyperparameter Tuning:** Prevented the model from overfitting the small dataset by constraining the trees (`max_depth=5`, `min_samples_split=10`).
* **Evaluation:** The tuned model improved accuracy to **~82%**.
* **Feature Importance:** Extracted the model's decision-making hierarchy, confirming historical realities:
  1. **Gender (`Sex_male`):** ~43.7% importance
  2. **Socio-Economic Status (`Fare` & `Pclass`):** ~30.6% combined importance
  3. **Age:** ~12.0% importance

## 🚀 Key Takeaways
The mathematical models perfectly mirrored historical accounts. The dominant predictors of survival were gender, passenger class, and age, confirming that the "women and children first" protocol and preferential treatment for higher-class passengers were the primary determinants of surviving the disaster.

## 🛠️ How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open the `notebooks/` directory in Jupyter Notebook or VS Code.
4. Run the notebooks in sequential order (01 -> 02 -> 03).

---