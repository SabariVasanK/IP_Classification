import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Read the processed dataset
data_file_location = "data/processed/"
data_file_name = "processed_data"
data_file_ext = ".csv"
data_df = pd.read_csv(data_file_location + data_file_name + data_file_ext, index_col=0)

# Separate independent and dependent variables
data_X = data_df.drop("label", axis=1)
data_y = data_df["label"]

# Identify categorical columns
categorical_cols = data_X.select_dtypes(include=['object']).columns.tolist()

# Encode categorical variables
ct = ColumnTransformer(
    [("encoder", OneHotEncoder(), categorical_cols)],
    remainder="passthrough"
)
data_X_encoded = ct.fit_transform(data_X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_X_encoded, data_y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
