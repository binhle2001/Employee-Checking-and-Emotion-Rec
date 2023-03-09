import pandas as pd
from settings import (
    DATA_FACE_DIR,
)


def main():
    df = pd.DataFrame(
        {
            "face": [],
            "name": [],
        }
    )
    df.to_json(DATA_FACE_DIR)


if __name__ == "__main__":
    main()
