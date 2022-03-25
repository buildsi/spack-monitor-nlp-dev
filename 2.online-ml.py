#!/usr/bin/env python

# Let's try doing kmeans with river!

from riverapi.main import Client
from river import cluster, feature_extraction, neighbors
from scipy.spatial.distance import pdist, squareform
from sklearn import manifold
from creme import feature_extraction as creme_features

import pandas
import argparse
import sys
import os
import pickle

sys.path.insert(0, os.getcwd())
from helpers import process_text, write_json, read_errors
from knn import KNeighborsClassifier


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


class ModelBuilder:
    def __init__(self, errors, host="http://127.0.0.1:8000", prefix=None, datadir=None):
        self.errors = errors
        self.datadir = datadir
        self.init_client(host, prefix)

    def init_client(self, host, prefix=None):
        """
        Create a river client (this is for development server, not yet spack monitor)
        """
        if prefix:
            self.cli = Client(host, prefix=prefix)
        else:
            self.cli = Client(host)

    def iter_sentences(self):
        """
        Yield sentences (parsed) to learn from.
        """
        for i, entry in enumerate(self.errors):
            print("%s of %s" % (i, len(self.errors)), end="\r")

            # NOTE if you change this, also change in 2.spack-issues-match.py
            # Pre, text, and post
            raw = entry.get("text")
            if not raw:
                continue

            # Split based on error
            if "error:" not in raw:
                continue

            text = raw.split("error:", 1)[-1]
            tokens = process_text(text)
            sentence = " ".join(tokens)

            # Skip single words!
            if not tokens or not sentence.strip() or len(tokens) == 1:
                continue
            yield sentence, entry["id"]

    def kmeans(self, model_name="spack-errors", save_prefix="kmeans"):
        """
        Build the kmeans model with a particular name.
        """
        # Create the model if it does not exist (I created beforehand)
        # Number clusters is kind of arbitrary, I wanted to do >> number of packages
        exists = True
        if model_name not in self.cli.models()["models"]:
            model = feature_extraction.BagOfWords() | cluster.KMeans(
                n_clusters=100, halflife=0.4, sigma=3, seed=0
            )
            model_name = self.cli.upload_model(model, "cluster", model_name=model_name)
            exists = False

        # Add each error to the server (only if not done yet)
        if not exists:
            for sentence, _ in self.iter_sentences():
                self.cli.learn(x=sentence, model_name=model_name)

            # Save clusters to file under data/clusters/<prefix>
            self.save_model(model_name)
            self.generate_clusters_json(model_name, save_prefix)

        return self.load_model("%s.pkl" % model_name)

    def knn(self, model_name="spack-knn-errors", save_prefix="knn"):
        """
        Build the knn model with a particular name.
        """
        model = creme_features.TFIDF() | KNeighborsClassifier(
            n_neighbors=5, window_size=10000, min_distance_keep=0.05
        )

        # Create a lookup of errors based on id so we can find quickly
        print("Creating errors lookup...")
        lookup = {}
        for sentence, uid in self.iter_sentences():
            lookup[uid] = sentence

        # I'm using the model directly since it takes an identifier
        print("Training KNN model with modified creme...")
        for sentence, uid in self.iter_sentences():
            model.fit_one(x=sentence, identifier=uid)

        # Save clusters to file under data/clusters/<prefix>
        cluster_dir = os.path.join(self.datadir, "clusters", save_prefix)
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        # Now get predictions
        print("Predictions...!\n")
        results = []
        result_file_number = 0
        count = 0
        for sentence, uid in self.iter_sentences():

            # Each neighbor has:
            # x, y, error id, and minkowski distnace
            neighbors = model.predict_one(x=sentence)
            neighbor_ids = [x[2] for x in neighbors]
            result = {
                "error": sentence,
                "error_id": uid,
                "neighbor_ids": neighbor_ids,
                "neighbors": [lookup[x] for x in neighbor_ids],
            }
            results.append(result)

            if count > 10000:
                result_meta = os.path.join(
                    cluster_dir, "errors-%s-neighbors.json" % result_file_number
                )
                write_json(results, result_meta)
                results = []
                count = 0
                result_file_number += 1
            else:
                count += 1

        # Save last to file
        if results:
            result_meta = os.path.join(
                cluster_dir, "errors-%s-neighbors.json" % result_file_number
            )
            write_json(results, result_meta)

        # Save model to file
        with open("%s.pkl" % model_name, "wb") as fd:
            pickle.dump(model, fd)
        return model

    def dbstream(self, model_name="spack-dbstream-errors", save_prefix="dbstream"):
        """
        Build the dbstream model with a particular name.
        https://riverml.xyz/latest/api/cluster/DBSTREAM
        """
        exists = True
        if model_name not in self.cli.models()["models"]:
            model = feature_extraction.BagOfWords() | cluster.DBSTREAM(
                clustering_threshold=1.5,
                fading_factor=0.05,
                cleanup_interval=4,
                intersection_factor=0.5,
                minimum_weight=1,
            )
            model_name = self.cli.upload_model(model, "cluster", model_name=model_name)
            exists = False

        if not exists:
            for sentence, _ in self.iter_sentences():
                self.cli.learn(x=sentence, model_name=model_name)

            # Save clusters to file under data/clusters/<prefix>
            self.generate_clusters_json(model_name, save_prefix)
            self.save_model(model_name)
        return self.load_model("%s.pkl" % model_name)

    def denstream(self, model_name="spack-dbstream-errors", save_prefix="denstream"):
        """
        Build the denstream model https://riverml.xyz/latest/api/cluster/DenStream/
        """
        # See https://github.com/online-ml/river/issues/874
        # model might have bugs! denstream I think is better
        # because denstream is good with outliers (we likely won't have)
        exists = True
        if model_name not in self.cli.models()["models"]:
            model = feature_extraction.BagOfWords() | cluster.DenStream(
                decaying_factor=0.01,
                beta=0.5,
                mu=2.5,
                epsilon=0.02,
            )
            model_name = self.cli.upload_model(model, "cluster", model_name=model_name)
            exists = False

        if not exists:
            for sentence, _ in self.iter_sentences():
                self.cli.learn(x=sentence, model_name=model_name)

            # Save clusters to file under data/clusters/<prefix>
            self.generate_clusters_json(model_name, save_prefix)
            self.save_model(model_name)
        return self.load_model("%s.pkl" % model_name)

    def generate_clusters_json(self, model_name, save_prefix):
        """
        Generate json cluster output for visual inspetion
        """
        # At this point, let's get a prediction for each
        # We can just group them based on the cluster
        clusters = {}
        for sentence, _ in self.iter_sentences():
            res = self.cli.predict(x=sentence, model_name=model_name)
            if res["prediction"] not in clusters:
                clusters[res["prediction"]] = []
            clusters[res["prediction"]].append(sentence)

        # Make model output directory
        cluster_dir = os.path.join(self.datadir, "clusters", save_prefix)
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        for cluster_id, entries in clusters.items():
            if not entries:
                continue
            cluster_meta = os.path.join(
                cluster_dir, "cluster-tokens-%s.json" % cluster_id
            )
            write_json(entries, cluster_meta)

    def save_model(self, model_name):
        """
        Save a pickled model (and return it)
        """
        # Download the model and load to get centroids
        self.cli.download_model(model_name=model_name)

    def load_model(self, pkl):
        """
        Load a model from pickle
        """
        with open(pkl, "rb") as fd:
            model = pickle.load(fd)
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

    if not os.path.exists("docs"):
        os.makedirs("docs")

    builder = ModelBuilder(datadir=datadir, errors=errors)

    # Get models to see if we have spack-errors
    # models = builder.cli.models()

    # Build knn model and export predictions
    # model = builder.knn(model_name="spack-knn-errors")

    # Build kmeans model and export clusters
    # Note that we don't need to keep doing that - spack-monitor can visualize them now
    # model = builder.kmeans(model_name="spack-errors")

    # Note that I removed saving embeddings here since we expect to do this on
    # spack monitor server.

    # model = builder.dbstream(model_name="spack-dbstream-errors")
    # model = builder.denstream(model_name="spack-denstream-errors")


if __name__ == "__main__":
    main()
