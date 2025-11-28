import argparse
from check_data_file_format import *
from check_correlations import validate_correlation_schema

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset correlations with pandera.")
    parser.add_argument(
        "--data",
        default="data/adult.csv",
        help="Path to CSV file to validate (default: data/adult.csv)",
    )
    file_path = parser.parse_args().data
    test_check_data_file_format(file_path)
    validate_correlation_schema(file_path)

if __name__ == "__main__":
    main()