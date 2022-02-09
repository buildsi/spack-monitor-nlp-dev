#!/usr/bin/env python

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from glob import glob
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

    # Split words with underscore into two words
    words = []
    for t in tokens:
        if "_" in t:
            words += [x.strip() for x in t.split("_")]

        # Don't add single letters
        elif len(t) == 1:
            continue
        else:
            words.append(t)

    # Don't do stemming here - the error messages are usually hard coded / consistent
    # words = [stemmer.stem(t) for t in tokens]
    return tokens


def write_json(content, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(content, indent=4))


def read_json(filename):
    with open(filename, "r") as fd:
        content = json.loads(fd.read())
    return content


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

    # Create a lookup of errors
    lookup_errors = {}
    lookup_parsed = {}

    # count undefined reference
    count = 0
    for entry in errors:

        # Pre, text, and post
        text = entry.get("text")
        if not text:
            continue

        if "undefined reference" in text:
            count += 1

        error = " ".join(process_text(text)).strip()
        if error not in lookup_errors:
            lookup_errors[error] = 0
        lookup_errors[error] += 1

        # Split based on error
        if "error:" not in text:
            continue

        text = text.split("error:", 1)[-1]
        text = " ".join(process_text(text)).strip()
        if text not in lookup_parsed:
            lookup_parsed[text] = 0
        lookup_parsed[text] += 1

    # Sort by value
    lookup_errors = {
        k: v
        for k, v in sorted(
            lookup_errors.items(), key=lambda item: item[1], reverse=True
        )
    }
    lookup_parsed = {
        k: v
        for k, v in sorted(
            lookup_parsed.items(), key=lambda item: item[1], reverse=True
        )
    }

    print("%s out of %s mention 'undefined reference'" %(count, len(errors)))
    write_json(lookup_parsed, os.path.join("docs", "parsed_errors_count.json"))
    write_json(lookup_errors, os.path.join("docs", "errors_count.json"))


if __name__ == "__main__":
    main()
