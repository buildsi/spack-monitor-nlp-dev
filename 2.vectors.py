#!/usr/bin/env python

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets, preprocessing, manifold

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
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
        description="Research Software Encyclopedia Preprocessor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        help="Directory with UID subfolders",
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

    # Since error output as filepaths, keep these
    text = text.replace(os.sep, " ")

    # Remove numbers and punctuation
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    # Strip whitespace
    text = text.strip()

    # tokenize and stop word removal
    tokens = [x for x in word_tokenize(text) if not x in stop_words]

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


def build_model(texts, name, outdir):

    # 40 epochs means we do it 40 times
    model = Doc2Vec(texts, vector_size=50, min_count=5, workers=4, epochs=40)

    # Save the model if we need again
    model.save(os.path.join(outdir, "model.%s.doc2vec" % name))

    # Create a vector for each document
    # UIDS as id for each row, vectors across columns
    df = pandas.DataFrame(columns=range(50))

    print("Generating vector matrix for documents...")
    for text in texts:
        df.loc[text.tags[0]] = model.infer_vector(text.words)

    # Save dataframe to file
    df.to_csv(os.path.join(outdir, "%s-vectors.csv" % name))

    # Create a distance matrix
    distance = pandas.DataFrame(
        squareform(pdist(df)), index=list(df.index), columns=list(df.index)
    )
    distance.to_csv(os.path.join(outdir, "%s-software-distances.csv" % name))

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

    errors = read_json(os.path.join(datadir, "errors.json"))
    warnings = read_json(os.path.join(datadir, "warnings.json"))

    # Make model output directory
    model_dir = os.path.join(datadir, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists("docs"):
        os.makedirs("docs")

    # Generate metadata for errors and warnings (lookup of ID)
    meta = {}
    for entry in errors + warnings:
        meta[entry["id"]] = entry
    write_json(meta, os.path.join("docs", "meta.json"))

    # Let's try creating three models: first pre, text, and post
    texts = []
    for entry in errors + warnings:

        # Pre, text, and post
        text = (
            entry.get("pre_context", "")
            + " "
            + entry.get("text")
            + " "
            + entry.get("post_context")
        )
        if not text:
            continue
        tokens = process_text(text)
        meta[entry["id"]] = entry
        texts.append(TaggedDocument(tokens, [entry["id"]]))

    build_model(texts, "pre-text-post", model_dir)

    # pre and text
    texts = []
    for entry in errors + warnings:

        # Pre, text
        text = entry.get("pre_context", "") + " " + entry.get("text")
        if not text:
            continue
        tokens = process_text(text)
        texts.append(TaggedDocument(tokens, [entry["id"]]))

    build_model(texts, "pre-text", model_dir)

    # post and text
    texts = []
    for entry in errors + warnings:

        # Pre, text
        text = entry.get("text") + " " + entry.get("post_context", "")
        if not text:
            continue
        tokens = process_text(text)
        texts.append(TaggedDocument(tokens, [entry["id"]]))

    build_model(texts, "text-post", model_dir)

    # now just text!
    texts = []
    for entry in errors + warnings:

        # Pre, text, and post
        text = entry.get("text")
        if not text:
            continue
        tokens = process_text(text)
        texts.append(TaggedDocument(tokens, [entry["id"]]))

    build_model(texts, "text", model_dir)


if __name__ == "__main__":
    main()
