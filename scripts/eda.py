import pandas as pd
import altair as alt
import click

alt.data_transformers.enable("vegafusion")

@click.command()
@click.option('--input_dir', required=True, help='Path (including filename) to training data')
def main(input_dir):
    #Input
    adult_df = pd.read_csv(input_dir)

    # Age Histogram 

    age_hist = alt.Chart(adult_df).mark_bar().encode(
        alt.X('age:Q', bin=alt.Bin(maxbins=20), title='Age'), 
        alt.Y('count():Q', title='Count'), 
        color='income:N', 
    ).properties(
        width=600, 
        height=400 
    )
    age_hist.save('age_hist.png')

    # Age Density Plot
    age_density = alt.Chart(adult_df).transform_density(
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

    age_density.save('age_density.png')

    # Marital Status - Frequency Chart
    marital_status_freq = alt.Chart(adult_df).mark_bar().encode(
        alt.X('count():Q', title='Count'),
        alt.Y('marital-status:N', title='Marital Status').sort('x')
    ).properties(
        width=300,  
        height=100 
    )

    marital_status_freq.save('marital_status_freq.png')

    # Race - Frequency Chart
    race_freq = alt.Chart(adult_df).mark_bar().encode(
        alt.X('count():Q', title='Count'),
        alt.Y('race:N', title='Race').sort('x')
    ).properties(
        width=300,  
        height=200 
    )

    race_freq.save('race_freq.png')

    # Education - Frequency Chart
    edu_freq = alt.Chart(adult_df).mark_bar().encode(
        x='count()',
        y=alt.Y('education', title='Education').sort('x')
    ).properties(
        width=300,  
        height=200 
    )

    edu_freq.save('edu_freq.png')

    # Work Class - Frequency Chart
    wc_freq = alt.Chart(adult_df).mark_bar().encode(
        x='count()',
        y=alt.Y('workclass', title='Work Class').sort('x')
    ).properties(
        width=300,  
        height=200 
    )

    edu_freq.save('edu_freq.png')


    # Native Country - Frequency Chart
    #adult_df['native-country'].value_counts()

    nc_freq = alt.Chart(adult_df).mark_bar().encode(
        x='count()',
        y=alt.Y('native-country', title='Native Country').sort('x')
    ).properties(
        width=300,  
        height=200 
    )

    nc_freq.save('nc_freq.png')

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

    corr_heatmap = (heatmap + labels).configure_axis(labelAngle=-45)

    corr_heatmap.save('corr_heatmap.png')

    # Prediction Class Distribution
    pred_class_dist = alt.Chart(adult_df).mark_bar().encode(
        alt.X('income:N', title='Income'),
        alt.Y('count():Q', title='Count'), 
        color='income:N' 
    ).properties(
        width=400, 
        height=300  
    )

    pred_class_dist.save('pred_class_dist.png')

if __name__ == '__main__':
    main()
