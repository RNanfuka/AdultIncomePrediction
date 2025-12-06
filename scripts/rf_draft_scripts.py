#adult_df.head(5)
#adult_df.tail(5)
#adult_df.shape
#adult_df.info()
#adult_df.describe().T.round(2)
#adult_df.nunique()

#adult_df.isnull().sum()

# for col in adult_df.columns:
#     print(f"Unique values in column '{col}':")
#     print(adult_df[col].unique())
#     print("-" * 20)



#adult_df.isnull().sum()

# Replace all ? categories across dataset with NAN
adult_df.replace('?', np.nan, inplace=True)

# Clean prediction values
adult_df['income'] = adult_df['income'].str.replace('<=50K.', '<=50K')
adult_df['income'] = adult_df['income'].str.replace('>50K.', '>50K')

# Feature Engineering, simplified categories for education column

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


# Feature Engineering, simplified categories for marital_status column

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

# - - - - - - - - - - - - - - - - - - - - START: EDA Split - - - - - - - - - - - - - - - - - - - -  

# Age Histogram 
alt.Chart(adult_df).mark_bar().encode(
    alt.X('age:Q', bin=alt.Bin(maxbins=20), title='Age'), 
    alt.Y('count():Q', title='Count'), 
    color='income:N', 
).properties(
    width=600, 
    height=400 
)


# Age Density Plot
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

# Marital Status - Frequency Chart
alt.Chart(adult_df).mark_bar().encode(
    alt.X('count():Q', title='Count'),
    alt.Y('marital-status:N', title='Marital Status').sort('x')
).properties(
    width=300,  
    height=100 
)

# Race - Frequency Chart
alt.Chart(adult_df).mark_bar().encode(
    alt.X('count():Q', title='Count'),
    alt.Y('race:N', title='Race').sort('x')
).properties(
    width=300,  
    height=200 
)

# Education - Frequency Chart
alt.Chart(adult_df).mark_bar().encode(
    x='count()',
    y=alt.Y('education', title='Education').sort('x')
).properties(
    width=300,  
    height=200 
)

# Work Class - Frequency Chart
alt.Chart(adult_df).mark_bar().encode(
    x='count()',
    y=alt.Y('workclass', title='Work Class').sort('x')
).properties(
    width=300,  
    height=200 
)


# Work Class - Frequency Chart
#adult_df['native-country'].value_counts()

alt.Chart(adult_df).mark_bar().encode(
    x='count()',
    y=alt.Y('native-country', title='Native Country').sort('x')
).properties(
    width=300,  
    height=200 
)

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

# Prediction Class Distribution
alt.Chart(adult_df).mark_bar().encode(
    alt.X('income:N', title='Income'),
    alt.Y('count():Q', title='Count'), 
    color='income:N' 
).properties(
    width=400, 
    height=300  
)
