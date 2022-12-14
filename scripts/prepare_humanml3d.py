import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pickle
import zipfile

from src.datamodules.components.humanml3d import HumanML3DProcessor

if not os.path.exists("data/HumanML3D/texts/"):
    print("Unzip annotation txts...")
    with zipfile.ZipFile("data/HumanML3D/texts.zip", "r") as zip_data:
        zip_data.extractall("data/HumanML3D/")
else:
    if not len(os.listdir("data/HumanML3D/texts")) == 29232:
        AssertionError("annotation files mismatch.")

with open("data/HumanML3D/humanact12/humanact12_processed.pkl", "rb") as f:
    humanact12_dict = pickle.load(f)


for split in ["train", "val", "test"]:
    print(f"Processing {split} set...")
    processor = HumanML3DProcessor(humanact12_dict, split=split)
    processor.process()

