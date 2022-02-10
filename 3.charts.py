#!/usr/bin/env python

from glob import glob
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

    print("%s out of %s mention 'undefined reference'" % (count, len(errors)))
    write_json(lookup_parsed, os.path.join("docs", "parsed_errors_count.json"))
    write_json(lookup_errors, os.path.join("docs", "errors_count.json"))


if __name__ == "__main__":
    main()
