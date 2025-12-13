import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import altair as alt
import click

from src.create_frequency_chart import create_frequency_chart

alt.data_transformers.enable("vegafusion")

@click.command()
@click.option('--input_dir', required=True, help='Path (including filename) to training data')
@click.option('--out_dir', required=True, help='Path to directory where the results should be saved')
def main(input_dir,out_dir):
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
    age_hist.save(out_dir+'age_hist.png')

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

    age_density.save(out_dir+'age_density.png')

    # Marital Status - Frequency Chart
    marital_status_freq = create_frequency_chart(adult_df, 'marital-status', height=100, title='Marital Status')
    marital_status_freq.save(out_dir + 'marital_status_freq.png')

    # Race - Frequency Chart
    race_freq = create_frequency_chart(adult_df, 'race', title='Race')
    race_freq.save(out_dir + 'race_freq.png')

    # Education - Frequency Chart
    edu_freq = create_frequency_chart(adult_df, 'education', title='Education')
    edu_freq.save(out_dir + 'edu_freq.png')

    # Work Class - Frequency Chart
    wc_freq = create_frequency_chart(adult_df, 'workclass', title='Work Class')
    wc_freq.save(out_dir + 'wc_freq.png')

    # Native Country - Frequency Chart
    nc_freq = create_frequency_chart(adult_df, 'native-country', width=400, height=600, title='Native Country')
    nc_freq.save(out_dir + 'nc_freq.png')

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

    corr_heatmap.save(out_dir+'corr_heatmap.png')

    # Prediction Class Distribution
    pred_class_dist = alt.Chart(adult_df).mark_bar().encode(
        alt.X('income:N', title='Income'),
        alt.Y('count():Q', title='Count'), 
        color='income:N' 
    ).properties(
        width=400, 
        height=300  
    )

    pred_class_dist.save(out_dir+'pred_class_dist.png')

if __name__ == '__main__':
    main()
