import pandas as pd
import pandera.pandas as pa

def test_data_content(df) -> None:
    schema = pa.DataFrameSchema(
        {
            "age": pa.Column(),
            "workclass": pa.Column(nullable=True),
            "fnlwgt": pa.Column(),
            "education": pa.Column(),
            "education-num": pa.Column(),
            "marital-status": pa.Column(),
            "occupation": pa.Column(nullable=True),
            "relationship": pa.Column(),
            "race": pa.Column(),
            "sex": pa.Column()
        },
        checks=[
            # Code adopted from https://ubc-dsci.github.io/reproducible-and-trustworthy-workflows-for-data-science/lectures/135-data_validation-python-pandera.html
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ]
    )

    schema.validate(df)
    print("Data content checks passed.")