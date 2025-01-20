Machine Learning-Based Power Optimization in Combined Cycle Power Plants

🔥 Project Overview

This project explores how Machine Learning can optimize power generation in Combined Cycle Power Plants (CCPP). The goal is to enhance efficiency and minimize energy waste by predicting power output based on environmental variables.

📊 Dataset

Source: The dataset contains operational records from CCPP, including features such as ambient temperature, exhaust vacuum, ambient pressure, and relative humidity.

File Used: Folds5x2_pp.ods

🔬 Key Insights and Results

Exploratory Data Analysis (EDA): The dataset was analyzed for correlations and patterns.

Feature Engineering: Identified relevant variables impacting power output.

Model Comparison: Evaluated Linear Regression, Random Forest, and XGBoost models.

Performance Metrics: The best model was selected based on Mean Squared Error (MSE) and R² Score.

Model

MSE

R² Score

Linear Regression

X.XX

X.XX

Random Forest

X.XX

X.XX

XGBoost

X.XX

X.XX

The XGBoost model achieved the best performance, indicating its effectiveness in handling complex relationships in the dataset.

📂 Project Structure

📂 Machine-Learning-Power-Optimization
│── 📜 README.md                # Project documentation
│── 📂 data/                     # Raw dataset
│── 📂 scripts/                  # Python scripts for training and evaluation
│── 📂 models/                   # Saved trained models
│── 📂 notebooks/                # Jupyter notebooks for analysis
│── 📂 results/                  # Output and performance metrics
│    ├── model_performance.txt   # Model evaluation metrics
│    ├── predictions.csv         # Predictions on new data
│    ├── feature_importance.png  # Feature importance visualization
│    ├── evaluation_plots.png    # Actual vs Predicted comparison
│── 📜 requirements.txt          # Dependencies list

🛠️ Technologies Used

Python 3.x

Pandas, NumPy (Data Handling)

Matplotlib, Seaborn (Visualization)

Scikit-Learn, XGBoost (Machine Learning)

Jupyter Notebook

🚀 How to Run

1️⃣ Clone the Repository

git clone https://github.com/your-username/Machine-Learning-Power-Optimization.git
cd Machine-Learning-Power-Optimization

2️⃣ Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

3️⃣ Run the Notebook

jupyter notebook notebooks/power_optimization.ipynb

4️⃣ Execute Scripts

python scripts/preprocess.py   # Data preprocessing
python scripts/train_model.py  # Train model
python scripts/evaluate.py     # Evaluate model
python scripts/predict.py      # Make predictions

📈 Results and Reports

results/model_performance.txt → Contains MSE and R² scores for model evaluation.

results/predictions.csv → Stores predictions made on new input data.

results/feature_importance.png → Shows the importance of different input features.

results/evaluation_plots.png → Scatter plot comparing actual vs. predicted power output.

📈 Conclusion

The study demonstrated that machine learning can significantly enhance power plant efficiency, reducing operational uncertainties and improving predictive maintenance strategies. Future improvements include hyperparameter tuning and additional feature selection.

📜 License

This project is licensed under the MIT License.

🤝 Contributions

Contributions are welcome! Feel free to fork this repository and submit pull requests for improvements.