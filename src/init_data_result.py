import pandas as pd
from settings import (
    DATA_RESULT_DIR,
)


def main():
    df = pd.DataFrame(
        {
            "name": [],
            "time": [],
        }
    )
    df.to_json(DATA_RESULT_DIR)


if __name__ == "__main__":
    main()
