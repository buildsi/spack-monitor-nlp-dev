#!/usr/bin/env python

# use the spack monitor API to download data and save locally.

import tempfile
import shutil
import requests
import argparse
import re
import json
import sys
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Spack Monitor Downloader",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        help="Output directory for data.",
        default=os.path.join(os.getcwd(), "data"),
    )
    return parser


def paginated_get(url):
    """
    Retrieve paginated results for an API endpoint.
    """
    results = []

    while True:
        print("Retrieving %s" % url)
        response = requests.get(url)
        if response.status_code != 200:
            sys.exit("Cannot get results for %s" % url)
        response = response.json()
        results = results + response.get("results", [])
        if "next" not in response or not response["next"]:
            break
        url = response["next"]
    return results


def write_json(content, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(content, indent=4))


def main():

    parser = get_parser()
    args, extra = parser.parse_known_args()

    # Make sure output directory exists
    outdir = os.path.abspath(args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get warnings from both instances of spack monitor!
    warns = paginated_get("https://builds.spack.io/api/buildwarnings/")
    warns = warns + paginated_get("https://monitor.spack.io/api/buildwarnings/")
    print("Found %s warnings" % len(warns))

    # And errors too!
    errors = paginated_get("https://builds.spack.io/api/builderrors/")
    errors = errors + paginated_get("https://monitor.spack.io/api/builderrors/")
    print("Found %s errors" % len(errors))

    write_json(errors, os.path.join("data", "errors.json"))
    write_json(warns, os.path.join("data", "warnings.json"))


if __name__ == "__main__":
    main()
