import pandas as pd
import click
from sklearn.model_selection import train_test_split # Clean labels and split early to avoid leakage in EDA

# Main function
@click.command()
@click.option('--input_dir', required=True, help='Path (including filename) to raw data')
@click.option('--train_out_dir', required=True, help='Path to directory where the train data should be saved')
@click.option('--test_out_dir', required=True, help='Path to directory where the test data should be saved')

def main(input_dir, train_out_dir, test_out_dir):
    #Input
    adult_df = pd.read_csv(input_dir)


    # Train/test split placed at the start of analysis
    data_train, data_test = train_test_split(
        adult_df,
        test_size = 0.8,
        random_state = 42,
        stratify = adult_df['income']
    )

    # Use training slice for EDA to avoid peeking at test data
    adult_df = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    #Output 
    adult_df.to_csv(train_out_dir + "train.csv", index=False) 
    
    data_test.to_csv(test_out_dir + "test.csv", index=False)

if __name__ == '__main__':
    main()


