# Import necessary libraries and modules
import pandas as pd
import numpy as np
import seaborn as sns
import ipaddress
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# Set display options
pd.set_option('display.max_columns', None)

# INITIALIZATION

# LOADING CLEANED DATASET

# Reading dataset file into pandas DataFrame
data_file_location = "data/interim/"
data_file_name =  "conn.log.labeled_cleaned.csv"

# Read the dataset
data_df = pd.read_csv(data_file_location + data_file_name, index_col=0)

# DATA PREPROCESSING

# Analyzing target attribute

# Check null values in the target attribute
data_df["label"].isna().sum()

# Check values distribution
data_df["label"].value_counts()

# Plot target attribute on a count plot
sns.countplot(data=data_df, x="label")

# Encoding target attribute
target_le = LabelEncoder()
encoded_attribute = target_le.fit_transform(data_df["label"])
data_df["label"] = encoded_attribute

# Handling outliers

# Use describe() method to obtain general statistics about the numerical features
numerical_features = ["duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
data_df[numerical_features].describe()

# Plot "duration" feature on a boxplot
data_df.reset_index(inplace=True)  # Reset index before creating the boxplot
sns.boxplot(data=data_df, y="duration")

# Replace outliers using IQR (Inter-quartile Range)
outliers_columns = ['duration']
for col_name in outliers_columns:
    q1, q3 = np.nanpercentile(data_df[col_name],[25,75])
    intr_qr = q3-q1
    iqr_min_val = q1-(1.5*intr_qr)
    iqr_max_val = q3+(1.5*intr_qr)
    data_df.loc[data_df[col_name] < iqr_min_val, col_name] = np.nan
    data_df.loc[data_df[col_name] > iqr_max_val, col_name] = np.nan

# Encoding IP addresses

def encode_ipv4(ip):
    return int(ipaddress.IPv4Address(ip))

data_df["id.orig_h"] = data_df["id.orig_h"].apply(encode_ipv4)
data_df["id.resp_h"] = data_df["id.resp_h"].apply(encode_ipv4)

# Handling missing values

data_df.isnull().sum().sort_values(ascending=False)

sns.heatmap(data=data_df.isnull(), yticklabels=False, cbar=False, cmap="viridis")

# Impute missing values: categorical features

sns.countplot(data=data_df, x="label", hue="service")

srv_training_columns = ["id.orig_h","id.orig_p","id.resp_h","id.resp_p","missed_bytes","orig_pkts","orig_ip_bytes","resp_pkts","resp_ip_bytes"] 
data_df_with_service = data_df[data_df["service"].notna()]
data_df_no_service = data_df[data_df["service"].isna()]

srv_X = data_df_with_service[srv_training_columns]
srv_y = data_df_with_service["service"].values

srv_X_train, srv_X_test, srv_y_train, srv_y_test = train_test_split(srv_X, srv_y, test_size=0.2, random_state=0)

srv_knn = KNeighborsClassifier(n_neighbors=3)
srv_knn.fit(srv_X_train, srv_y_train)

srv_y_pred = srv_knn.predict(srv_X_test)

srv_accuracy_test = accuracy_score(srv_y_test, srv_y_pred)
print(f"Prediction accuracy for 'service' is: {srv_accuracy_test}")
print("Classification report:")
print(classification_report(srv_y_test, srv_y_pred))

srv_predictions = srv_knn.predict(data_df_no_service[srv_training_columns])

data_df.loc[data_df["service"].isna(), "service"] = srv_predictions

# Impute missing values: numerical features

numerical_features = data_df.drop("label", axis=1).select_dtypes(include="number").columns
knn_imputer = KNNImputer()
data_df_after_imputing = knn_imputer.fit_transform(data_df[numerical_features])
data_df[numerical_features] = data_df_after_imputing

# Scaling numerical attributes

numerical_features = ["id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "duration", "orig_bytes", "resp_bytes", "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]
scaler = MinMaxScaler()
data_df[numerical_features] = scaler.fit_transform(data_df[numerical_features])

# Save the processed dataset
processed_data_file_location = "data/processed/"
processed_data_file_name = "processed_data"
processed_data_file_ext = ".csv"

data_df.to_csv(processed_data_file_location + processed_data_file_name + processed_data_file_ext)
