from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocessing():
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
    return preprocessor
