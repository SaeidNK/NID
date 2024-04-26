import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json
from preprocess import preprocessing
from Train import Train


# Load the dataset
df = pd.read_csv('./archive/Train_data.csv')
"""
# Define your target variable (the value you want to predict)
target = 'class' 
# Define preprocessing for categorical columns (encode them)
categorical_features = ['protocol_type', 'service', 'flag']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Define preprocessing for numerical columns (scale them)
numerical_features = ['src_bytes', 'dst_bytes']
numerical_transformer = StandardScaler()

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])
"""
preprocessor = preprocessing()
# Create instances of the classifiers
logistic_regression = LogisticRegression()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()

# Create an array containing the classifiers
classifiers = [logistic_regression, decision_tree, random_forest]
for clf in classifiers:
    print(Train(df, preprocessor, clf))
"""

# Create a pipeline that combines the preprocessor with a classifier
print('---------------------------------LogisticRegression')
clf1 = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

# Split the data into features (X) and target variable (y)
X = df.drop(columns=[target])
y = df[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline (including preprocessing) to the training data
clf1.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf1.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
"""

"""
#-------------------------------------------------------------saving model
joblib.dump(clf_dt, 'model.pkl')
#Json format
# Convert train and test data splits to JSON
train_json = {
    "X_train": X_train.to_json(orient="records"),
    "y_train": y_train.to_json(orient="records")
}

test_json = {
    "X_test": X_test.to_json(orient="records"),
    "y_test": y_test.to_json(orient="records")
}

# Specify file paths for saving JSON files
train_file_path = "train_data.json"
test_file_path = "test_data.json"

# Save train and test data splits to JSON files
with open(train_file_path, "w") as train_file:
    train_file.write(json.dumps(train_json, indent=4))

with open(test_file_path, "w") as test_file:
    test_file.write(json.dumps(test_json, indent=4))

print("Train and test data splits saved to JSON files.")
"""