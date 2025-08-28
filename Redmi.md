Customer Churn Prediction  

🚀 Features  
 Data preprocessing with pipelines (handling missing values, encoding, scaling).  
 Addressed class imbalance using SMOTE (Synthetic Minority Oversampling Technique).  
 Trained and evaluated multiple machine learning models (Logistic Regression, Decision Trees, Random Forest, etc.).  
 Compared model performance to select the most accurate one.  
 Generated classification reports, accuracy scores, and visualization plots.  

🛠️ Tech Stack  
Python
Pandas, NumPy– Data processing  
Scikit-learn – ML models & pipelines  
Imbalanced-learn – SMOTE  
Matplotlib / Seaborn – Data visualization  

 📂 Project Structure  

structure of machine leaning project
│── data/                    # Raw and processed data (avoid uploading large raw datasets)
│   ├── raw/                      # Original raw dataset (or link inREADME)
│   └── processed/                # Cleaned datasets
│
│── notebooks/             # Jupyter notebooks for exploration (EDA, experiments)
│   ├── 01_EDA.ipynb
│   └── 02_Model_Baseline.ipynb
│
│── src/                          # Source code (modularized)
│   ├── data_preprocessing.py     # Data loading & cleaning functions
│   ├── features.py               # Feature engineering
│   ├── model.py                  # Model building (pipeline, training)
│   └── evaluate.py               # Metrics, evaluation
│
│── models/                       # Saved models/pipelines
│   └── Cutomer_Churn_prediction.pkl
│
│── train.py                  # Main training script (uses src modules + pipe)
│── predict.py                # Script to load trained model & predict on new data
│── requirements.txt              # Dependencies (scikit-learn, pandas, etc.)
│── README.md                     # Project documentation
│── .gitignore                    # Ignore cache, venv, big data files



