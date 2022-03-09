#!/usr/bin/env python

# Let's try doing kmeans with river!

from riverapi.main import Client
from river import synth, evaluate, metrics, neighbors
from river import cluster, feature_extraction
from scipy.spatial.distance import pdist, squareform
from sklearn import manifold

from glob import glob
import pandas
import argparse
import sys
import os
import pickle

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
        print("%s of %s" % (i, len(errors)), end="\r")
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
            sentence = " ".join(tokens)
            if not tokens or not sentence.strip():
                continue
            yield sentence


def build_dbstream(cli, errors, model_name, datadir):
    """
    Build the dbstream model with a particular name.
    https://riverml.xyz/latest/api/cluster/DBSTREAM
    """
    exists = True
    if model_name not in cli.models()["models"]:
        model = feature_extraction.BagOfWords() | cluster.DBSTREAM(
            clustering_threshold=1.5,
            fading_factor=0.05,
            cleanup_interval=4,
            intersection_factor=0.5,
            minimum_weight=1,
        )
        model_name = cli.upload_model(model, "cluster", model_name=model_name)
        exists = False

    if not exists:
        for sentence in iter_sentences(errors):
            res = cli.learn(x=sentence, model_name=model_name)


def build_kmeans(cli, errors, model_name, datadir):
    """
    Build the kmeans model with a particular name.
    """
    # Create the model if it does not exist (I created beforehand)
    # Number clusters is kind of arbitrary, I wanted to do >> number of packages
    exists = True
    if model_name not in cli.models()["models"]:
        model = feature_extraction.BagOfWords() | cluster.KMeans(
            n_clusters=100, halflife=0.4, sigma=3, seed=0
        )
        model_name = cli.upload_model(model, "cluster", model_name=model_name)
        exists = False

    # Add each error to the server (only if not done yet)
    if not exists:
        for sentence in iter_sentences(errors):
            res = cli.learn(x=sentence, model_name=model_name)

    # At this point, let's get a prediction for each
    # We can just group them based on the cluster
    clusters = {}
    for sentence in iter_sentences(errors):
        res = cli.predict(x=sentence, model_name=model_name)
        if res["prediction"] not in clusters:
            clusters[res["prediction"]] = []
        clusters[res["prediction"]].append(sentence)

    # Make model output directory
    cluster_dir = os.path.join(datadir, "clusters", "kmeans")
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

    for cluster_id, entries in clusters.items():
        if not entries:
            continue
        cluster_meta = os.path.join(cluster_dir, "cluster-tokens-%s.json" % cluster_id)
        write_json(entries, cluster_meta)

    # Download the model and load to get centroids
    cli.download_model(model_name=model_name)
    with open("spack-errors.pkl", "rb") as fd:
        model = pickle.load(fd)

    # Save the centers and create data for visualization because people expect it at this point :/
    return model


def generate_embeddings(centers, name):
    df = pandas.DataFrame(centers)

    # 200 rows (centers) and N columns (words)
    df = df.transpose()
    df = df.fillna(0)

    # Create a distance matrix
    distance = pandas.DataFrame(
        squareform(pdist(df)), index=list(df.index), columns=list(df.index)
    )

    # Make the tsne (output embeddings go into docs for visual)
    fit = manifold.TSNE(n_components=2)
    embedding = fit.fit_transform(distance)
    emb = pandas.DataFrame(embedding, index=distance.index, columns=["x", "y"])
    emb.index.name = "name"
    emb.to_csv(os.path.join("docs", "%s-software-embeddings.csv" % name))


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
    # models = cli.models()

    # Build kmeans model and export clusters
    # Note that we don't need to keep doing that - spack-monitor can visualize them now
    # model = build_kmeans(cli=cli, errors=errors, model_name="spack-errors", datadir=datadir)

    # Save the centers and generate distance matrix
    # centers = model.steps["KMeans"].centers
    # write_json(centers, os.path.join(cluster_dir, "centers.json"))

    # UIDS as id for each row, vectors across columns
    # centers = list(centers.values())
    # generate_embeddings(centers, "kmeans")
    model = build_dbstream(
        cli=cli, errors=errors, model_name="spack-dbstream-errors", datadir=datadir
    )


if __name__ == "__main__":
    main()
