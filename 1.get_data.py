#!/usr/bin/env python

# use the spack monitor API to download data and save locally.

import requests
import argparse
import sys
import os

sys.path.insert(0, os.getcwd())
from helpers import write_json


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
    url = "%s?limit=10000" % url
    while True:
        print("Retrieving %s" % url)
        try:
            response = requests.get(url)
        except Exception as e:
            print(e)
            yield []
        if response.status_code != 200:
            sys.exit("Cannot get results for %s" % url)
        response = response.json()
        if "next" not in response or not response["next"]:
            break
        url = response["next"]
        results = response.get("results", [])
        if not results:
            break
        yield results


def main():

    parser = get_parser()
    args, extra = parser.parse_known_args()

    # Make sure output directory exists
    outdir = os.path.abspath(args.outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get warnings from both instances of spack monitor!
    for url in [
        "https://builds.spack.io/api/buildwarnings/",
        "https://monitor.spack.io/api/buildwarnings/",
    ]:
        for i, results in enumerate(paginated_get(url)):
            for result in results:
                result["spack-monitor-label"] = "warning"
            if not results:
                break
            write_json(results, os.path.join("data", "warnings-%s.json" % i))

    # And errors too!
    for url in [
        "https://builds.spack.io/api/builderrors/",
        "https://monitor.spack.io/api/builderrors/",
    ]:
        for i, results in enumerate(paginated_get(url)):
            for result in results:
                result["spack-monitor-label"] = "error"
            if not results:
                break
            write_json(results, os.path.join("data", "errors-%s.json" % i))


if __name__ == "__main__":
    main()
