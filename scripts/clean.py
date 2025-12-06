
import pandas as pd
import numpy as np

# Main function
@click.command()
@click.option('--input_dir', required=True, help='Path (including filename) to raw data')
@click.option('--out_dir', required=True, help='Path to directory where the results should be saved')
def main(input_dir, out_dir):

    #Input
    adult_df = pd.read_csv(input_dir)

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

    #Output 
    adult_df.to_csv(out_dir, index=False)

if __name__ == '__main__':
    main()
