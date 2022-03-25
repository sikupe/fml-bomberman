import argparse
import sys
import logging
import os
import numpy as np

parser = argparse.ArgumentParser(description="Merge multiple trained models")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser.add_argument(
    "path", type=str, help="Directory containing the trained models"
)

parser.add_argument(
    "destination", type=str, help="Optional path of the output file", nargs='?'
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
        destination = os.path.join(os.getcwd(), "merged_model.npy")

    model_files = [file for file in os.listdir(abs_path) if ".npy" in file]
    models = [np.load(os.path.join(abs_path, file)) for file in model_files]
    merged_model = np.mean(np.array(models), axis=0)
    np.save(destination, merged_model)
    logger.info("Saved model to %s", destination)
