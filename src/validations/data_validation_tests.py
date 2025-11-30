import argparse

import pandas as pd

from check_correlations import get_tests as get_correlation_tests
from check_data_content import get_tests as get_data_content_tests
from check_data_file_format import get_tests as get_file_format_tests


def run_tests(tests):
    """Execute each test callable and return a list of result dictionaries."""
    results = []
    for name, func in tests:
        try:
            passed = func()
            status = "PASS" if passed is not False else "FAIL"
            error = "" if status == "PASS" else "Returned False"
        except Exception as exc:  # pylint: disable=broad-except
            status = "FAIL"
            error = str(exc)
        results.append({"test": name, "status": status, "error": error})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset correlations with pandera.")
    parser.add_argument(
        "--data",
        default="data/adult.csv",
        help="Path to CSV file to validate (default: data/adult.csv)",
    )
    parser.add_argument(
        "--no-raise-errors",
        action="store_true",
        help="Suppress exceptions inside individual tests; failures return False instead.",
    )
    args = parser.parse_args()
    file_path = args.data
    df = pd.read_csv(file_path)
    raise_on_fail = not args.no_raise_errors

    tests = []
    tests.extend(get_file_format_tests(file_path, raise_on_fail=raise_on_fail))
    tests.extend(get_data_content_tests(df, raise_on_fail=raise_on_fail))
    tests.extend(get_correlation_tests(df, raise_on_fail=raise_on_fail))

    results = run_tests(tests)

    table = pd.DataFrame(results, columns=["test", "status", "error"])
    print("**** Validation Results ****")
    print(table.to_string(index=False, justify="left"))

    if any(result["status"] == "FAIL" for result in results):
        raise SystemExit(1)
    print("All validations passed.")


if __name__ == "__main__":
    main()
