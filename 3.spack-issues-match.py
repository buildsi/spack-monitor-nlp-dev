#!/usr/bin/env python

# Let's download all of our spack issues.

from github import Github
import os
import argparse
from glob import glob
import sys

sys.path.insert(0, os.getcwd())
from helpers import read_json, process_text, write_json, read_errors


def read_issues(datadir):
    for filename in glob(os.path.join(datadir, "issue-*.json")):
        yield read_json(filename)


def parse_text(raw):
    if not raw:
        return

    # Split based on error
    if "error:" not in raw:
        return

    # We already have error: in these lines
    text = raw.split("error:", 1)[-1]
    tokens = process_text(text)
    sentence = " ".join(tokens)

    # Skip single words!
    if not tokens or not sentence.strip() or len(tokens) == 1:
        return
    return sentence.strip()


def match_issues(result, datadir):
    """
    Read in our issues and look for matches.
    """
    errors = read_errors(datadir)
    parsed = set()
    for entry in errors:
        text = entry.get("text")
        error = parse_text(text)
        if not error:
            continue
        parsed.add(error)

    found = {}
    print("Found %s unique errors in the spack monitor set." % len(parsed))
    for number, entry in result["errors"].items():
        for error in entry["parsed"]:
            if error in parsed:
                if number not in found:
                    found[number] = []
                found[number].append(error)
        if number in found:
            found[number] = list(set(found[number]))
    return found


def parse_issues(issues_dir):
    """
    Find spack issues that have errors in them.
    """
    errors = {}
    no_match = []
    for issue in read_issues(issues_dir):

        # Look for error: in the body:
        if not issue["body"] or "error:" not in issue["body"]:
            no_match.append(issue["number"])
        else:
            lines = [x for x in issue["body"].split("\n") if "error:" in x]
            parsed = [parse_text(x) for x in lines if parse_text(x)]
            errors[issue["number"]] = {"raw": lines, "parsed": parsed}
    return errors, no_match


def get_parser():
    parser = argparse.ArgumentParser(
        description="Spack Issue Matcher",
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

    issues_dir = os.path.join(datadir, "spack-issues")
    if not os.path.exists(issues_dir):
        sys.exit("%s does not exist!" % issues_dir)

    # Get errors found, and list of issues without match
    outfile = os.path.join(datadir, "spack-issue-errors.json")
    if not os.path.exists(outfile):
        errors, no_match = parse_issues(issues_dir)
        print(
            "Found %s issues with errors, %s issues without matches."
            % (len(errors), len(no_match))
        )
        result = {
            "errors": errors,
            "total": len(no_match) + len(errors),
            "no_match": len(no_match),
        }
        write_json(result, outfile)
    else:
        result = read_json(outfile)

    # Now match!
    found = match_issues(result, datadir)
    outfile = os.path.join(datadir, "spack-issue-errors-found-spack-monitor.json")
    write_json(found, outfile)


if __name__ == "__main__":
    main()
