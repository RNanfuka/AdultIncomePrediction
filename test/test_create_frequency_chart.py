import pytest
import pandas as pd
import altair as alt

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.create_frequency_chart import create_frequency_chart

@pytest.fixture
def sample_df():
    """Create a sample dataframe for testing."""
    return pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C'],
        'numeric': [1, 2, 3, 4, 5, 6, 7, 8],
        'text': ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple', 'cherry', 'cherry']
    })

def test_returns_altair_chart(sample_df):
    """Test that function returns an Altair Chart object."""
    chart = create_frequency_chart(sample_df, 'category')

    #Checks that altair chart is created
    assert isinstance(chart, alt.Chart)

def test_default_parameters(sample_df):
    """Test chart creation with default parameters."""
    chart = create_frequency_chart(sample_df, 'category')
    
    # Check chart properties
    assert chart.width == 300
    assert chart.height == 200
    assert chart.mark == 'bar'

def test_custom_dimensions(sample_df):
    """Test chart creation with custom width and height."""
    chart = create_frequency_chart(sample_df, 'category', width=500, height=400)
    
    assert chart.width == 500
    assert chart.height == 400