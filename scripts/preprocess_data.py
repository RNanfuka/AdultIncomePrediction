import pandas as pd
import click

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_feature_names(preprocessor):
    feature_names = []

    # Ordinal features
    ordinal_features = preprocessor.transformers_[0][2]
    feature_names.extend(ordinal_features)

    # Binary features
    binary_features = preprocessor.transformers_[1][2]
    binary_encoder = preprocessor.transformers_[1][1].named_steps['encoder']
    binary_feature_names = binary_encoder.get_feature_names_out(binary_features)
    feature_names.extend(binary_feature_names)

    # Categorical features
    categorical_features = preprocessor.transformers_[2][2]
    categorical_encoder = preprocessor.transformers_[2][1].named_steps['encoder']
    categorical_feature_names = categorical_encoder.get_feature_names_out(categorical_features)
    feature_names.extend(categorical_feature_names)

    # Numeric features
    numeric_features = preprocessor.transformers_[3][2]
    feature_names.extend(numeric_features)

    return feature_names

# Main function
@click.command()
@click.option('--input_dir', required=True, help='Path (including filename) to raw data')
@click.option('--out_dir', required=True, help='Path to directory where the preprocessed_adult_train.csv should be saved')
def main(input_dir, out_dir):
    #Input
    adult_df = pd.read_csv(input_dir)

    # Separate features and target using the pre-created train/test split
    numeric_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'native-country']
    ordinal_features = ['education']
    binary_features = ['sex']
    drop_features = ['fnlwgt', 'education-num', 'race']
    target = "income"

    X_train = adult_df.drop(columns = drop_features + [target])
    y_train = adult_df[target]

    # Ordered levels for the education column (lowest to highest)
    education_order = [
        'dropout',
        'HighGrad',
        'CommunityCollege',
        'Bachelors',
        'Masters',
        'Doctorate',
    ]

    # Column Preprocessing
    # Pipelines for different data types
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    binary_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
    ])

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_pipeline, ordinal_features),
            ('binary', binary_pipeline, binary_features),
            ('categorical', categorical_pipeline, categorical_features),
            ('numeric', numeric_pipeline, numeric_features),
        ]
    )

    # generate preprocessed training data just to check how it looks
    X_train_processed = preprocessor.fit_transform(X_train)

    # get the feature names after preprocessing
    feature_names = get_feature_names(preprocessor)

    # convert processed data back to DataFrame for better readability
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_train_processed_df[target] = y_train.reset_index(drop=True)

    #Output
    if "train" in input_dir:
        X_train_processed_df.to_csv(out_dir + "preprocessed_train.csv", index=False)
        click.echo(f"Saved preprocessed_train.csv to file directory {out_dir}")
    elif "test" in  input_dir:
        X_train_processed_df.to_csv(out_dir + "preprocessed_test.csv", index=False)
        click.echo(f"Saved preprocessed_test.csv to file directory {out_dir}")
    else:
        X_train_processed_df.to_csv(out_dir + "preprocessed_data.csv", index=False)
        click.echo(f"Saved preprocessed_data.csv to file directory {out_dir}")

if __name__ == '__main__':
    main()
