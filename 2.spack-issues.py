#!/usr/bin/env python

# Let's download all of our spack issues.

from github import Github
import os
import argparse
import sys

sys.path.insert(0, os.getcwd())
from helpers import write_json


def get_issues(issues_dir):
    g1 = Github(token)

    org = g1.get_organization("spack")
    repo = org.get_repo("spack")

    issues = repo.get_issues()
    for issue in issues:
        outfile = os.path.join(issues_dir, "issue-%s.json" % issue.number)
        if os.path.exists(outfile):
            continue
        print("Parsing issue %s" % issue.number)
        meta = {
            "body": issue.body,
            "user": issue.user.login,
            "url": issue.url,
            "updated_at": str(issue.updated_at),
            "created_at": str(issue.created_at),
            "closed_at": str(issue.closed_at),
            "state": issue.state,
            "title": issue.title,
            "number": issue.number,
            "milestone": issue.milestone,
            "labels": [x.name for x in issue.labels],
            "id": issue.id,
            "html_url": issue.html_url,
            "assignees": [x.login for x in issue.assignees],
            "comments": issue.comments,
        }
        write_json(meta, outfile)


token = os.environ.get("GITHUB_TOKEN")
if not token:
    sys.exit("Yo dawg this is a LOT of API calls, you really need a GITHUB_TOKEN")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Spack Issue Downloaded",
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
        os.makedirs(issues_dir)

    # Save as we go
    get_issues(issues_dir)


if __name__ == "__main__":
    main()
