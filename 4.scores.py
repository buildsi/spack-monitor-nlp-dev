#!/usr/bin/env python

from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
from glob import glob
import statistics
import argparse
import sys
import os

sys.path.insert(0, os.getcwd())
from helpers import process_text, write_json, read_json


def get_parser():
    parser = argparse.ArgumentParser(
        description="Spack Monitor Analyzer",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        help="Directory with data",
        default=os.path.join(os.getcwd(), "data"),
    )
    return parser


def main():

    parser = get_parser()
    args, extra = parser.parse_known_args()

    # Make sure output directory exists
    datadir = os.path.abspath(args.data_dir)
    if not os.path.exists(datadir):
        sys.exit("%s does not exist!" % datadir)

    # Build model with errors
    errors = []
    for filename in glob(os.path.join(datadir, "errors*.json")):
        errors += read_json(filename)
    print("Found %s errors!" % len(errors))

    # Load in model
    model = Doc2Vec.load(os.path.join("data", "models", "model.error.doc2vec"))

    scores = []
    # for each error, calculate homogeneity score
    for entry in errors:

        # Pre, text, and post
        text = entry.get("text")
        if not text:
            continue

        # Split based on error
        if "error:" not in text:
            continue

        text = text.split("error:", 1)[-1]
        tokens = process_text(text)
        new_vector = model.infer_vector(tokens)
        sims = model.docvecs.most_similar([new_vector])

        # NOT a perfect metric, take mean and sd
        nums = [x[1] for x in sims]
        scores.append((statistics.mean(nums), statistics.stdev(nums)))

    # We can save these if needed
    means = [s[0] for s in scores]
    stdevs = [s[0] for s in scores]

    plt.hist(means, bins=100)
    plt.title("KNN with N=10, average similarity for 30K messages")
    plt.savefig(os.path.join("data", "means.png"))


if __name__ == "__main__":
    main()
