# Load Data In

from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
alt.data_transformers.enable("vegafusion")

# fetch dataset 
adult = fetch_ucirepo(id=2) 
# data (as pandas dataframes)  
X = pd.DataFrame(adult.data.features) 
y = pd.DataFrame(adult.data.targets) 
adult_df = pd.concat([X, y], axis = 1)
# persist combined dataset to CSV
adult_df.to_csv("data/adult.csv", index=False)

# read dataset from CSV
adult_df = pd.read_csv("data/adult.csv")
adult_df.shape

# Clean labels and split early to avoid leakage in EDA
from sklearn.model_selection import train_test_split

# Train/test split placed at the start of analysis
data_train, data_test = train_test_split(
    adult_df,
    test_size=0.8,
    random_state=42,
    stratify=adult_df['income']
)
# Use training slice for EDA to avoid peeking at test data
adult_df = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)

adult_df.head(5)
adult_df.tail(5)
adult_df.shape
adult_df.info()
adult_df.describe().T.round(2)
adult_df.nunique()
adult_df.isnull().sum()
for col in adult_df.columns:
    print(f"Unique values in column '{col}':")
    print(adult_df[col].unique())
    print("-" * 20)
adult_df.replace('?', np.nan, inplace=True)
adult_df.isnull().sum()
adult_df['income'] = adult_df['income'].str.replace('<=50K.', '<=50K')
adult_df['income'] = adult_df['income'].str.replace('>50K.', '>50K')

education_mapping = {
    'Preschool': 'dropout',
    '10th': 'dropout',
    '11th': 'dropout',
    '12th': 'dropout',
    '1st-4th': 'dropout',
    '5th-6th': 'dropout',
    '7th-8th': 'dropout',
    '9th': 'dropout',
    'HS-Grad': 'HighGrad',
    'HS-grad': 'HighGrad',
    'Some-college': 'CommunityCollege',
    'Assoc-acdm': 'CommunityCollege',
    'Assoc-voc': 'CommunityCollege',
    'Masters': 'Masters',
    'Prof-school': 'Masters',
}

adult_df['education'] = adult_df['education'].replace(education_mapping)

marital_status_mapping = {
    'Never-married': 'NotMarried',
    'Married-AF-spouse': 'Married',
    'Married-civ-spouse': 'Married',
    'Married-spouse-absent': 'NotMarried',
    'Separated': 'Separated',
    'Divorced': 'Separated',
    'Widowed': 'Widowed'
}

adult_df['marital-status'] = adult_df['marital-status'].replace(marital_status_mapping)

# EDA
alt.Chart(adult_df).mark_bar().encode(
    alt.X('age:Q', bin=alt.Bin(maxbins=20), title='Age'), 
    alt.Y('count():Q', title='Count'), 
    color='income:N', 
).properties(
    width=600, 
    height=400 
)

alt.Chart(adult_df).transform_density(
     'age',
     groupby=['income'],
     as_=['age', 'density'],
).mark_area(
     opacity=0.4
 ).encode(
     x='age',
     y=alt.Y('density:Q').stack(False),
     color='income'
)

alt.Chart(adult_df).mark_bar().encode(
    alt.X('count():Q', title='Count'),
    alt.Y('marital-status:N', title='Marital Status').sort('x')
).properties(
    width=300,  
    height=100 
)

alt.Chart(adult_df).mark_bar().encode(
    alt.X('count():Q', title='Count'),
    alt.Y('race:N', title='Race').sort('x')
).properties(
    width=300,  
    height=200 
)

alt.Chart(adult_df).mark_bar().encode(
    x='count()',
    y=alt.Y('education', title='Education').sort('x')
).properties(
    width=300,  
    height=200 
)

alt.Chart(adult_df).mark_bar().encode(
    x='count()',
    y=alt.Y('workclass', title='Work Class').sort('x')
).properties(
    width=300,  
    height=200 
)

adult_df['native-country'].value_counts()
# alt.Chart(adult_df).mark_point(opacity=0.6, size=2).encode(
#     alt.X(alt.repeat('row')).type('quantitative'),
#     alt.Y(alt.repeat('column')).type('quantitative'),
#     color='income'  # Map 'prediction' to color (nominal type)
# ).properties(
#     width=130,
#     height= 130
# ).repeat(
#     column=['age','education-num','capital-gain','capital-loss','hours-per-week'],
#     row=['age', 'education-num','capital-gain','capital-loss','hours-per-week']
# )

# Altair correlation heatmap for numeric features
corr = adult_df.corr(numeric_only=True)
numeric_cols = corr.columns.tolist()

corr_long = corr.stack().reset_index()
corr_long.columns = ['var1', 'var2', 'corr']

heatmap = alt.Chart(corr_long).mark_rect().encode(
    x=alt.X('var1:N', sort=numeric_cols, title=''),
    y=alt.Y('var2:N', sort=list(reversed(numeric_cols)), title=''),
    color=alt.Color('corr:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1]), title='Correlation'),
    tooltip=[alt.Tooltip('var1:N', title='Feature 1'),
             alt.Tooltip('var2:N', title='Feature 2'),
             alt.Tooltip('corr:Q', title='Correlation', format='.2f')]
).properties(width=480, height=480)

labels = alt.Chart(corr_long).mark_text(baseline='middle', color='black', size=12).encode(
    x=alt.X('var1:N', sort=numeric_cols),
    y=alt.Y('var2:N', sort=list(reversed(numeric_cols))),
    text=alt.Text('corr:Q', format='.2f')
)

(heatmap + labels).configure_axis(labelAngle=-45)

alt.Chart(adult_df).mark_bar().encode(
    alt.X('income:N', title='Income'),
    alt.Y('count():Q', title='Count'), 
    color='income:N' 
).properties(
    width=400, 
    height=300  
)

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Separate features and target using the pre-created train/test split
numeric_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'native-country']
ordinal_features = ['education']
binary_features = ['sex']
drop_features = ['fnlwgt', 'education-num', 'race']
target = "income"

# Ordered levels for the education column (lowest to highest)
education_order = [
    'dropout',
    'HighGrad',
    'CommunityCollege',
    'Bachelors',
    'Masters',
    'Doctorate',
]

X_train = adult_df.drop(columns=drop_features + [target])
X_test = data_test.drop(columns=drop_features + [target])
y_train = adult_df[target]
y_test = data_test[target]

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

preprocessor
# generate preprocessed training data just to check how it looks
X_train_processed = preprocessor.fit_transform(X_train)

# get the feature names after preprocessing
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

feature_names = get_feature_names(preprocessor)

# convert processed data back to DataFrame for better readability
X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_train_processed_df[target] = y_train.reset_index(drop=True)
X_train_processed_df.to_csv("data/preprocessed_adult_train.csv", index=False)
# Modeling and result summarization moved to scripts/modeling.py.
