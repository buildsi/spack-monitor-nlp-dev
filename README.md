# Spack Monitor NLP

This is an effort to see if we can do some kind of clustering using the warning and error messages in
the server. The goal will be to:

1. retrieve current warnings and errors via the spack monitor API
2. build a word2vec model using them.
3. output embeddings for each
4. cluster!

## Usage

```bash
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

Then download data from spack monitor

```bash
$ python 1.get_data.py
```

This will generate a file of errors and warnings!

```bash
$ tree data/
data/
├── errors.json
└── warnings.json
```

We next want to preprocess the data and generate models / vectors!

```bash
$ python 2.vectors.py
```

Some data will be generated in data, and assets for the web interface will go
into [docs](docs). The interface allows you to select and see the difference between
the models, and clearly just using the error message has the strongest signal (best clustering).
Unforunately the database is strongly weighed toward boost errors, so the clusters
reflect that.

## License

Spack is distributed under the terms of both the MIT license and the
Apache License (Version 2.0). Users may choose either license, at their
option.

All new contributions must be made under both the MIT and Apache-2.0
licenses.

See [LICENSE-MIT](https://github.com/spack/spack/blob/develop/LICENSE-MIT),
[LICENSE-APACHE](https://github.com/spack/spack/blob/develop/LICENSE-APACHE),
[COPYRIGHT](https://github.com/spack/spack/blob/develop/COPYRIGHT), and
[NOTICE](https://github.com/spack/spack/blob/develop/NOTICE) for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

LLNL-CODE-811652
