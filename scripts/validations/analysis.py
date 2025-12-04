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
adult_df.to_csv("../data/adult.csv", index=False)

# read dataset from CSV
adult_df = pd.read_csv("../data/adult.csv")
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
X_train_processed_df.head()

# Baseline and Model comparison
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

# Code adapted from MDS 571 course materials
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)

models = {
    'Dummy-most_frequent': Pipeline([
        ('preprocess', preprocessor),
        ('classifier', DummyClassifier(strategy='most_frequent'))
    ]),
    'DecisionTree': Pipeline([
        ('preprocess', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ]),
    'KNN': Pipeline([
        ('preprocess', preprocessor),
        ('classifier', KNeighborsClassifier())
    ]),
    'SVM-RBF': Pipeline([
        ('preprocess', preprocessor),
        ('classifier', SVC(kernel='rbf'))
    ]),
    'LogisticRegression': Pipeline([
        ('preprocess', preprocessor),
        ('classifier', LogisticRegression(max_iter=200))
    ]),
    'GaussianNB': Pipeline([
        ('preprocess', preprocessor),
        ('classifier', GaussianNB())
    ]),
}

cv_results = {}
for name, model in models.items():
    cv_results[name] = mean_std_cross_val_scores(
        model, X_train, y_train, cv=5, return_train_score=True
    )

cv_summary = pd.DataFrame(cv_results).T
cv_summary

# Hyperparameter tuning:
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

log_reg_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=500, random_state=42))
])

log_reg_params = {
    'classifier__C': loguniform(1e-3, 1e3),
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs'],
}

log_reg_search = RandomizedSearchCV(
    log_reg_pipe,
    param_distributions=log_reg_params,
    n_iter=200,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=1,
)

log_reg_search.fit(X_train, y_train)

print(f"Best Logistic Regression CV accuracy: {log_reg_search.best_score_:.3f}")
print("Best params:", log_reg_search.best_params_)

svm_pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', SVC(kernel='rbf', random_state=42))
])

svm_params = {
    'classifier__C': loguniform(1e-2, 1e2),
    'classifier__gamma': loguniform(1e-4, 1e0),
}

svm_search = RandomizedSearchCV(
    svm_pipe,
    param_distributions=svm_params,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=1,
)

svm_search.fit(X_train, y_train)

print(f"Best SVM (RBF) CV accuracy: {svm_search.best_score_:.3f}")
print("Best params:", svm_search.best_params_)

# Use tuned models when available; otherwise fall back to the quick baseline fits
best_log_reg = log_reg_search.best_estimator_
log_reg_name = "Tuned Logistic Regression"

best_svm = svm_search.best_estimator_
svm_name = "Tuned RBF-SVM"


for name, model in [
    (log_reg_name, best_log_reg),
    (svm_name, best_svm),
]:
    test_acc = model.score(X_test, y_test)
    print(f"{name} test accuracy: {test_acc:.3f}")

# Model comparison and interpretation:
import numpy as np
import pandas as pd

log_reg_clf = best_log_reg.named_steps['classifier']
feature_names = best_log_reg.named_steps['preprocess'].get_feature_names_out()
coefs = log_reg_clf.coef_[0]

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefs,
    'odds_ratio': np.exp(coefs)
})

# Strongest positive and negative signals
top_positive = coef_df.sort_values('coefficient', ascending=False).head(10)
top_negative = coef_df.sort_values('coefficient').head(10)

print("Top positive predictors (increase odds of >50K):")
display(top_positive)

print("Top negative predictors (decrease odds of >50K):")
display(top_negative)

# visual check for interpretability
import altair as alt
import pandas as pd

# Altair bars for top coefficients
pos_chart = (
    alt.Chart(top_positive)
    .mark_bar(color='seagreen')
    .encode(
        y=alt.Y('feature:N', sort=list(top_positive['feature'][::-1]), title='Feature'),
        x=alt.X('coefficient:Q', title='Coefficient (log-odds)'),
        tooltip=['feature', 'coefficient', 'odds_ratio']
    )
    .properties(title='Top positive coefficients', width=280, height=230)
)

neg_chart = (
    alt.Chart(top_negative)
    .mark_bar(color='firebrick')
    .encode(
        y=alt.Y('feature:N', sort=list(top_negative['feature'][::-1]), title='Feature'),
        x=alt.X('coefficient:Q', title='Coefficient (log-odds)'),
        tooltip=['feature', 'coefficient', 'odds_ratio']
    )
    .properties(title='Top negative coefficients', width=280, height=230)
)

(pos_chart | neg_chart)
