import operator

from creme import base


__all__ = ["KNeighborsRegressor", "KNeighborsClassifier"]


def minkowski_distance(a: dict, b: dict, p: int):
    """Minkowski distance.
    Parameters:
        a
        b
        p: Parameter for the Minkowski distance. When `p=1`, this is equivalent to using the
            Manhattan distance. When `p=2`, this is equivalent to using the Euclidean distance.
    """
    return sum(
        (abs(a.get(k, 0.0) - b.get(k, 0.0))) ** p for k in set([*a.keys(), *b.keys()])
    )


class NearestNeighbours:
    def __init__(self, window_size, p, min_distance_keep=0.05):
        self.window_size = window_size
        self.p = p
        self.window = []

        # Quick lookup of errors to skip distances if needed
        self.lookup = set()
        self.min_distance_keep = min_distance_keep

    def update(self, x, y, identifier, k):

        # Don't add VERY similar points to window
        # Euclidean distance min is
        nearest = self.find_nearest(x, k)

        # Quick way to skip calculating distances
        key = " ".join(x)
        if key in self.lookup:
            return self

        # If we have any points too similar, don't keep
        distances = [x[3] for x in nearest if x[3] < self.min_distance_keep]
        if not distances:
            # Have we gone over the length and need to remove one?
            if len(self.window) >= self.window_size:
                print("Removing from window %s: %s" % (identifier, key))
                removed = self.window.pop(0)
                self.lookup.remove(" ".join(removed[0]))

            # Add a new x, y, and identifier to the window and lookup
            print("Adding to window %s: %s" % (identifier, " ".join(x)))
            self.window.append((x, y, identifier))
            self.lookup.add(" ".join(" ".join(x)))
        return self

    def find_nearest(self, x, k):
        """Returns the `k` closest points to `x`, along with their distances."""

        # Compute the distances to each point in the window
        points = ((*p, minkowski_distance(a=x, b=p[0], p=self.p)) for p in self.window)

        # Return the k closest points, index 3 is the distance (preceeded by x,y,identifier)
        return sorted(points, key=operator.itemgetter(3))[:k]


class KNeighborsClassifier(base.Classifier):
    """K-Nearest Neighbors (KNN) for classification.

    This works by storing a buffer with the `window_size` most recent observations. A brute-force
    search is used to find the `n_neighbors` nearest observations in the buffer to make a
    prediction.

    Parameters:
        n_neighbors: Number of neighbors to use.
        window_size: Size of the sliding window use to search neighbors with.
        p: Power parameter for the Minkowski metric. When `p=1`, this corresponds to the
            Manhattan distance, while `p=2` corresponds to the Euclidean distance.
        weighted: Whether to weight the contribution of each neighbor by it's inverse
            distance or not.

    Example:

        >>> from creme import datasets
        >>> from creme import evaluate
        >>> from creme import metrics
        >>> from creme import neighbors
        >>> from creme import preprocessing

        >>> dataset = datasets.Phishing()

        >>> model = (
        ...     preprocessing.StandardScaler() |
        ...     neighbors.KNeighborsClassifier()
        ... )

        >>> metric = metrics.Accuracy()

        >>> evaluate.progressive_val_score(dataset, model, metric)
        Accuracy: 84.55%

    """

    def __init__(self, n_neighbors=5, window_size=50, p=2, min_distance_keep=0.05):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.p = p
        self.classes = set()
        self._nn = NearestNeighbours(
            window_size=window_size, p=p, min_distance_keep=min_distance_keep
        )

    @property
    def _multiclass(self):
        return True

    def fit_one(self, x, y, identifier):
        self.classes.add(y)
        self._nn.update(x, y, identifier, k=self.n_neighbors)
        return self

    def predict_one(self, x: dict):
        """Predict the label of a set of features `x`.
        Parameters:
            x: A dictionary of features.
        Returns:
            The neighbors
        """
        return self.predict_proba_one(x)

    def predict_proba_one(self, x):
        """
        This is modified to just return the nearest, not try to calculate
        a prediction because we just want the points back.
        """
        return self._nn.find_nearest(x=x, k=self.n_neighbors)
