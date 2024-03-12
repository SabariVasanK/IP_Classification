# Import necessary libraries and modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import precision_score, confusion_matrix, recall_score, accuracy_score, f1_score
from statistics import mean
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from joblib import dump

# Set display options
pd.set_option('display.max_columns', None)

# LOADING PROCESSED DATASET

# Initialize required variables to read the cleaned data file
data_file_location = "data/processed/"
data_file_name = "processed_data"
data_file_ext = ".csv"

# Read the dataset
data_df = pd.read_csv(data_file_location + data_file_name + data_file_ext, index_col=0)

# Exploring dataset summary and statistics
print("Dataset shape:", data_df.shape)
print("Dataset head:", data_df.head())

# Split data into independent and dependent variables
data_df_encoded = pd.get_dummies(data_df)
data_X = data_df_encoded.drop("label", axis=1)
data_y = data_df_encoded["label"]

# MODEL TRAINING

print("Model Training Started!")

# Initialize classification models
classifiers = [
    ("Naive Bayes", ComplementNB()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Logistic Regression", LogisticRegression()),
    ("Random Forest", RandomForestClassifier()),
    ("Support Vector Classifier", SVC()),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("XGBoost", xgb.XGBClassifier(objective="binary:logistic", alpha=10))
]

# Initialize the cross-validator with 5 splits and sample shuffling activated
skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Initialize the results summary
classification_results = pd.DataFrame(index=[c[0] for c in classifiers], columns=["Accuracy", "TN", "FP", "FN", "TP", "Recall", "Precision", "F1"])

# Iterate over the estimators
for est_name, est_object in classifiers:
    print(f"### [{est_name}]: Processing ...")
    
    # Initialize the results for each classifier
    accuracy_scores = []
    confusion_matrices = []
    recall_scores = []
    precision_scores = []
    f1_scores = []
    
    # Iterate over the obtained folds
    for train_index, test_index in skf_cv.split(data_X, data_y):
        # Get train and test samples from the cross-validation model
        X_train, X_test = data_X.iloc[train_index], data_X.iloc[test_index]
        y_train, y_test = data_y.iloc[train_index], data_y.iloc[test_index]
        
        # Train the model
        est_object.fit(X_train.values, y_train.values)
        
        # Predict the test samples
        y_pred = est_object.predict(X_test.values)
        
        # Calculate and register accuracy metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        est_f1_score = f1_score(y_test, y_pred)
        f1_scores.append(est_f1_score)
        
    # Summarize the results for all folds for each classifier
    tn, fp, fn, tp = sum(confusion_matrices).ravel()
    classification_results.loc[est_name] = [mean(accuracy_scores), tn, fp, fn, tp, mean(recall_scores), mean(precision_scores), mean(f1_scores)]
    
    # Save the best performing model
    model_name = est_name.replace(' ', '_').replace('-', '_').lower()
    model_file = model_name + ".pkl"
    dump(est_object, model_file)

print("Model Training Finished!")

# Check the results
print(classification_results)
