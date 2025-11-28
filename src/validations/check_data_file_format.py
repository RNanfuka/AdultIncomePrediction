import os

def test_check_data_file_format(file_path) -> None:
    # Adopted Copilot's answer to "Can the `pandera` python package check the file format?"
    if not file_path.endswith(".csv"):
        raise ValueError("File must be a CSV")
    print("Data File Format Tests passed.")