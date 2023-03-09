import os
import shutil
import sys
import pandas as pd

df = pd.read_json("Dataset/FaceData/processed/data_face.json")


def main(name):
    members = df[df["name"] != name]
    members.to_json("Dataset/FaceData/processed/data_face.json")
    shutil.rmtree("Dataset/FaceData/raw/" + name)
    # os.remove('Dataset/FaceData/raw/' + name)


if __name__ == "__main__":
    name = sys.argv[1]
    sys.exit(main(name) or 0)
