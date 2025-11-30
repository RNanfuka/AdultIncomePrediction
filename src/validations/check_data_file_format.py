import os


def test_file_exists(file_path: str, raise_on_fail: bool = True) -> bool:
    """Return True if the dataset path exists; optionally raise on failure."""
    exists = os.path.exists(file_path)
    if not exists and raise_on_fail:
        raise ValueError(f"File does not exist: {file_path}")
    return exists


def test_check_data_file_format(file_path: str, raise_on_fail: bool = True) -> bool:
    """Return True if the dataset path points to a CSV; optionally raise on failure."""
    is_csv = file_path.endswith(".csv")
    if not is_csv and raise_on_fail:
        raise ValueError("File must be a CSV")
    return is_csv


def get_tests(file_path: str, raise_on_fail: bool = True):
    """Return a list of (name, callable) tests for file format validation."""
    return [
        ("file path exists", lambda: test_file_exists(file_path, raise_on_fail)),
        ("file is csv", lambda: test_check_data_file_format(file_path, raise_on_fail)),
    ]
