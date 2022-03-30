#!/usr/bin/env python3
from tqdm import tqdm
import sys
import numpy as np
import os
import argparse
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Extract number per line and drop as np array")
parser.add_argument("input", type=str, help="input list file")

parser.add_argument(
    "destination", type=str, help="output file"
)

if __name__ == "__main__":
    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    destination = os.path.abspath(args.destination)

    if not os.path.isfile(input_file):
        logger.error("Input file is no file")
        sys.exit(1)

    if not os.path.isfile(destination):
        logger.error("Destination already exists")
        sys.exit(1)

    with open("./data.list", "r") as f:
        res = np.zeros((10000))
        with open("combined-python.list", "w+") as w:
            for i, line in tqdm(enumerate(f)):
                res[i] = sum([int(e) for e in line.split(",")])
            np.save("./arr.npy", res)

