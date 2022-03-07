#!/usr/bin/env python

# Let's try doing kmeans with river!

from riverapi.main import Client
from river import synth, evaluate, metrics, neighbors
from river import cluster, feature_extraction

from glob import glob
import pandas
import umap
import argparse
import sys
import os

sys.path.insert(0, os.getcwd())
from helpers import process_text, write_json, read_json, read_errors


def get_parser():
    parser = argparse.ArgumentParser(
        description="Spack Monitor Online ML",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        help="Directory with data",
        default=os.path.join(os.getcwd(), "data"),
    )
    return parser


def iter_sentences(errors, return_text=False, return_raw=False):
    for i, entry in enumerate(errors):
        print("%s of %s" %(i, len(errors)), end='\r')
        # Pre, text, and post
        raw = entry.get("text")
        if not raw:
            continue

        # Split based on error
        if "error:" not in raw:
            continue

        text = raw.split("error:", 1)[-1]
        if return_text:
            yield text
        elif return_raw:
            yield raw
        else:
            tokens = process_text(text)
            sentence = ' '.join(tokens)
            if not tokens or not sentence.strip():
                continue
            yield sentence


def main():

    parser = get_parser()
    args, extra = parser.parse_known_args()

    # Make sure output directory exists
    datadir = os.path.abspath(args.data_dir)
    if not os.path.exists(datadir):
        sys.exit("%s does not exist!" % datadir)

    # Build model with errors
    errors = read_errors(datadir)

    # Make model output directory
    model_dir = os.path.join(datadir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists("docs"):
        os.makedirs("docs")
    
    # Create a river client (this is for development server, not yet spack monitor)
    # cli = Client("http://127.0.0.1", prefix="ml")
    cli = Client("http://127.0.0.1:8000")

    # Get models to see if we have spack-errors
    models = cli.models()

    # Add this here so we don't run stupidly
    import IPython
    IPython.embed()
    sys.exit()

    # Create the model if it does not exist (I created beforehand)
    # Number clusters is kind of arbitrary, I wanted to do >> number of packages
    model_name = 'spack-errors'
    if model_name not in cli.models['models']:
        model = feature_extraction.BagOfWords() | cluster.KMeans(n_clusters=100, halflife=0.4, sigma=3, seed=0)
        model_name = cli.upload_model(model, "cluster", model_name=model_name)

    # model_name = 'spack-nearest-errors'
    # model = feature_extraction.BagOfWords() | neighbors.KNNADWINClassifier(window_size=100)
    # model_name = cli.upload_model(model, "cluster", model_name=model_name)
        
    # Add each error to the server
    for sentence in iter_sentences(errors)
        res = cli.learn(x=sentence, model_name=model_name)

    # At this point, let's get a prediction for each
    # We can just group them based on the cluster
    clusters = {}
    for sentence in iter_sentences(errors)
        res = cli.predict(x=sentence, model_name=model_name)
        if res['prediction'] not in clusters:
            clusters[res['prediction']] = []
        clusters[res['prediction']].append(entry)

    # Make model output directory
    cluster_dir = os.path.join(datadir, "clusters")
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    
    for cluster_id, entries in clusters.items():
        if not entries:
            continue

        # Make a simplified version of just tokens
        sentences = list(iter_sentences(entries, return_text=True))        

        cluster_meta = os.path.join(cluster_dir, "cluster-%s.json" % cluster_id)
        write_json(entries, cluster_meta)
        cluster_meta = os.path.join(cluster_dir, "cluster-tokens-%s.json" % cluster_id)
        write_json(sentences, cluster_meta)


if __name__ == "__main__":
    main()
