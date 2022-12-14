import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pprint import pprint
from src.datamodules.components.babel import BABELProcessor, BABELSummarizer
import yaml

def prepare_dataset(config):
    # print preparation configs
    pprint(config)

    target_fps = config["target_fps"]
    if config["prepare_babel"]:
        print("Process BABEL...")
        babel_motion_length = config["babel_motion_length"]
        babel_offset = config["babel_offset"]

        # Process BABEL Dataset first.
        for babel_file in ["train", "extra_train", "val", "extra_val"]:
            babel_summarizer = BABELSummarizer(
                babel_dir="data/babel_v1.0_release", babel_file=babel_file, act_cat_only=False
            )
            babel_summarizer.save_extract_summary()
        print(f"BABEL summary is saved at [{babel_summarizer.babel_summary_dir}]")

        print("Process BABEL training data...")
        babel_train_processor = BABELProcessor(
            summary_dir="data/babel_summary",
            summary_files=["train.csv", "extra_train.csv"],
            amass_dir="data/amass_smplhg",
            split="train",
            motion_length=babel_motion_length,
            target_fps=target_fps,
            offset=babel_offset,
            suffix=config["suffix"],
            labels=config["labels"],
        )
        babel_train_processor.process()

        print("Process BABEL validation data...")
        babel_val_processor = BABELProcessor(
            summary_dir="data/babel_summary",
            summary_files=["val.csv", "extra_val.csv"],
            amass_dir="data/amass_smplhg",
            split="val",
            motion_length=babel_motion_length,
            target_fps=target_fps,
            offset=babel_offset,
            suffix=config["suffix"],
            labels=config["labels"],
        )
        babel_val_processor.process()
        print("BABEL process complete.")


if __name__ == "__main__":

    with open("configs/prepare_babel.yaml") as f:
        config = yaml.safe_load(f)

    prepare_dataset(config)
