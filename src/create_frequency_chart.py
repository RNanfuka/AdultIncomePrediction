import pandas as pd
import altair as alt

alt.data_transformers.enable("vegafusion")

def create_frequency_chart(df, column, width=300, height=200, title=None, sort_order='-x'):
    """
    Create a horizontal frequency bar chart for a categorical column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to create frequency chart for
    width : int, default 300
        Width of the chart in pixels
    height : int, default 200
        Height of the chart in pixels
    title : str, optional
        Y-axis title. If None, uses the column name
    sort_order : str, default '-x'
        Sort order: 'x' for ascending, '-x' for descending
    
    Returns
    -------
    alt.Chart
        Altair chart object
    """
    if title is None:
        title = column
    
    chart = alt.Chart(df).mark_bar().encode(
        x='count()',
        y=alt.Y(f'{column}:N', title=title).sort(sort_order)
    ).properties(
        width=width,
        height=height
    )
    
    return chart