import random

def random_test_record(len: int) -> int:
    """
    Selects a random test record from the test dataset and returns its index.
    """
    total_test_records = len
    test_record = random.randint(0, total_test_records - 1)
    return test_record