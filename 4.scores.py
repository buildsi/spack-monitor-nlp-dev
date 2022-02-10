#!/usr/bin/env python

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from glob import glob
import statistics
import pandas
import tempfile
import shutil
import argparse
import json
import re
import sys
import os


# Derive stop words and stemmer once
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


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


def write_json(content, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(content, indent=4))


def read_json(filename):
    with open(filename, "r") as fd:
        content = json.loads(fd.read())
    return content


def process_text(text):
    """
    Process text, including:

    1. Lowercase
    2. Remove numbers and punctuation
    3. Strip whitespace
    4. Tokenize and stop word removal
    5. Stemming
    """
    # Make lowercase
    text = text.lower()

    # Remove numbers and punctuation (but leave path separator for now)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s\/]", "", text)

    # Strip whitespace
    text = text.strip()

    # tokenize and stop word removal
    tokens = [x for x in word_tokenize(text) if not x in stop_words]

    # Since error output as filepaths get rid of paths!
    # Get rid of anything that looks like a path!
    tokens = [x for x in tokens if os.sep not in x]

    # Don't do stemming here - the error messages are usually hard coded / consistent
    # words = [stemmer.stem(t) for t in tokens]
    return tokens


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
