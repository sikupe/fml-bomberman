#!/usr/bin/env python3
import argparse
import sys
import logging
import os
import numpy as np

parser = argparse.ArgumentParser(description="Merge multiple trained models")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser.add_argument("path", type=str, help="Directory containing the trained models")

parser.add_argument(
    "destination", type=str, help="Optional path of the output file", nargs="?"
)


if __name__ == "__main__":
    args = parser.parse_args()

    abs_path = os.path.abspath(args.path)

    if not os.path.isdir(abs_path):
        logger.error("Path is no directory")
        sys.exit(1)

    if args.destination:
        destination = args.destination
    else:
        destination = os.path.join(os.getcwd(), "merged_model")

    for file_extension in [".npy", ".pt"]:
        model_files = [file for file in os.listdir(abs_path) if file_extension in file]
        if len(model_files) == 0:
            continue
        models = [np.load(os.path.join(abs_path, file)) for file in model_files]
        merged_model = np.mean(np.array(models), axis=0)
        # If user supplied destination, don't add file_extension
        model_file = destination if args.destination else destination + file_extension
        np.save(model_file, merged_model)
        print(
            f"{file_extension}: Saved model to {model_file}. Model shape {merged_model.shape}."
        )
