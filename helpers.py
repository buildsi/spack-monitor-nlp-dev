from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import json
import re
import os

# Derive stop words and stemmer once
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def write_json(content, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(content, indent=4))


def read_json(filename):
    with open(filename, "r") as fd:
        content = json.loads(fd.read())
    return content


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
    words = []
    for token in word_tokenize(text):
        if token in stop_words:
            continue
        if os.sep in token:
            token = os.path.basename(token)
        words.append(token)

    # Don't do stemming here - the error messages are usually hard coded / consistent
    # words = [stemmer.stem(t) for t in tokens]
    return words
