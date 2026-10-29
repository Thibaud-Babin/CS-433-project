"""Data loading and preprocessing for the BRFSS heart disease dataset.

Steps (all statistics are learned on the training set only):
1. drop survey metadata columns and columns with too many NaN
2. columns with few distinct values are one-hot encoded
3. the other columns are cleaned (special BRFSS codes), NaN are replaced by
   the median and outliers are clipped
4. collinear columns are removed, then everything is standardized and a
   bias column of ones is added
"""

import os

import numpy as np

from helpers import load_csv_data

SELECTED_FEATURES = [
    "_RFHYPE5",
    "TOLDHI2",
    "_CHOLCHK",
    "_BMI5",
    "SMOKE100",
    "CVDSTRK3",
    "DIABETE3",
    "_TOTINDA",
    "_FRTLT1",
    "_VEGLT1",
    "_RFDRHV5",
    "HLTHPLN1",
    "MEDCOST",
    "GENHLTH",
    "MENTHLTH",
    "PHYSHLTH",
    "DIFFWALK",
    "SEX",
    "_AGEG5YR",
    "EDUCA",
    "INCOME2",
]


METADATA_COLUMNS = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "LADULT",
    "CTELNUM1",
    "CELLFON2",
    "CADULT",
    "PVTRESD2",
    "CCLGHOUS",
    "CSTATE",
    "LANDLINE",
    "QSTVER",
    "QSTLANG",
    "MSCODE",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALCOR",
    "_DUALUSE",
    "_LLCPWT",
    "_IMPRACE",
    "_CHISPNC",
    "_CRACE1",
    "_CPRACE",
]

CACHE_NAMES = ["x_train", "x_test", "y_train", "train_ids", "test_ids"]


def load_data(data_dir="data"):
    """Load the dataset.

    Reading the csv files is slow (a few minutes), so the first time we save
    the arrays as .npy files in data/cache/ and load those the next times.

    Returns x_train, x_test, y_train (in {-1, 1}), train_ids, test_ids and
    the list of column names.
    """
    cache_dir = os.path.join(data_dir, "cache")
    paths = []
    for name in CACHE_NAMES:
        paths.append(os.path.join(cache_dir, name + ".npy"))

    cache_exists = True
    for path in paths:
        if not os.path.exists(path):
            cache_exists = False

    if not cache_exists:
        os.makedirs(cache_dir, exist_ok=True)
        arrays = load_csv_data(data_dir)
        for i in range(len(paths)):
            np.save(paths[i], arrays[i])

    x_train = np.load(paths[0], mmap_mode="r")
    x_test = np.load(paths[1], mmap_mode="r")
    y_train = np.load(paths[2])
    train_ids = np.load(paths[3])
    test_ids = np.load(paths[4])

    with open(os.path.join(data_dir, "x_train.csv")) as f:
        header = f.readline().strip().split(",")
    columns = []
    for name in header[1:]:
        columns.append(name.strip('"'))
    return x_train, x_test, y_train, train_ids, test_ids, columns


def to_01(y):
    """Convert labels from {-1, 1} to {0, 1}."""
    y = np.array(y)
    return ((y + 1) / 2).astype(np.float32)


def to_pm1(y):
    """Convert labels from {0, 1} (or booleans) to {-1, 1} for the submission."""
    y = np.array(y, dtype=int)
    return 2 * y - 1


def repeated_digit(digit, n_digits):
    """Number made of n_digits times the same digit, e.g. (7, 3) -> 777."""
    return int(str(digit) * n_digits)


def clean_sentinels(col, col_max=None):
    """Replace the special BRFSS codes of a continuous column.

    BRFSS uses repeated digits for special answers, e.g. for a 2 digit
    question: 77 = don't know, 99 = refused (-> NaN) and 88 = none (-> 0).
    The number of digits is found from the largest value of the column.
    For the test set, col_max must be the max of the training column, so
    that both sets are cleaned the same way.

    Returns a cleaned copy of the column.
    """
    col = np.array(col, dtype=np.float64)
    if col_max is None:
        if np.all(np.isnan(col)):
            return col
        col_max = np.nanmax(col)

    if col_max == 900 or col_max == 99900:
        col[col == col_max] = np.nan
        return col

    if col_max != int(col_max) or col_max < 77:
        return col

    n_digits = len(str(int(col_max)))
    dont_know = repeated_digit(7, n_digits)
    refused = repeated_digit(9, n_digits)
    none = repeated_digit(8, n_digits)
    if col_max != dont_know and col_max != refused:
        return col

    col[col == dont_know] = np.nan
    col[col == refused] = np.nan
    col[col == none] = 0
    if n_digits == 3:
        col[col == 555] = 0
    return col


def find_collinear_columns(z, tol=1e-7):
    """Find the columns of z that are (almost) a linear combination of the
    columns before them. They make tx^T tx singular, which breaks least
    squares (e.g. two questions skipped by the same people give the same
    NaN indicator).

    We use the QR decomposition z = QR: R[j, j]^2 is the squared norm of the
    part of column j that is not explained by the columns before it. z is
    too big for memory in float64, so R is computed chunk by chunk:
    the R of [R_previous; next chunk] is the R of all rows seen so far.

    Returns a boolean array, True for the columns to keep.
    """
    n_rows = z.shape[0]
    n_cols = z.shape[1]
    R = np.zeros((0, n_cols))
    squared_norms = np.zeros(n_cols)
    chunk_size = 20000
    for start in range(0, n_rows, chunk_size):
        chunk = z[start : start + chunk_size].astype(np.float64)
        squared_norms += np.sum(chunk**2, axis=0)
        R = np.asarray(np.linalg.qr(np.vstack([R, chunk]), mode="r"))

    keep = np.ones(n_cols, dtype=bool)
    for j in range(n_cols):
        if squared_norms[j] == 0 or j >= R.shape[0]:
            keep[j] = False
            continue
        residual = R[j, j] ** 2 / squared_norms[j]
        if residual <= tol:
            keep[j] = False
    return keep


class Preprocessor:
    """Learns the preprocessing on the training set (fit) and applies it to
    any data (transform).

    Parameters:
        features: list of columns to use, or None for all columns except
            METADATA_COLUMNS
        max_nan_frac: columns with more NaN than this fraction are dropped
        max_categories: columns with at most this many distinct values are
            one-hot encoded
        min_category_frac: categories (and NaN indicators) are only kept if
            at least this fraction of the samples has them
        clip_quantile: continuous columns are clipped to the
            [q, 1 - q] quantiles of the training set
    """

    def __init__(
        self,
        features=None,
        max_nan_frac=0.8,
        max_categories=15,
        min_category_frac=0.005,
        clip_quantile=0.001,
    ):
        self.features = features
        self.max_nan_frac = max_nan_frac
        self.max_categories = max_categories
        self.min_category_frac = min_category_frac
        self.clip_quantile = clip_quantile

    def fit(self, x, columns):
        """Learn the features to build and their statistics from the
        training data x (columns = list of column names of x)."""
        if self.features is None:
            names = []
            for c in columns:
                if c not in METADATA_COLUMNS:
                    names.append(c)
        else:
            names = list(self.features)
        min_count = self.min_category_frac * x.shape[0]

        self.specs_ = []
        self.feature_names_ = ["bias"]
        for name in names:
            index = columns.index(name)
            raw = np.array(x[:, index], dtype=np.float64)
            is_nan = np.isnan(raw)
            if np.mean(is_nan) > self.max_nan_frac:
                continue
            values, counts = np.unique(raw[~is_nan], return_counts=True)

            if len(values) <= self.max_categories:
                reference = values[np.argmax(counts)]
                levels = []
                for i in range(len(values)):
                    if counts[i] >= min_count and values[i] != reference:
                        levels.append(values[i])
                spec = {
                    "kind": "categorical",
                    "index": index,
                    "levels": levels,
                    "nan_indicator": np.sum(is_nan) >= min_count,
                }
                for v in levels:
                    self.feature_names_.append(f"{name}=={v:g}")
            else:
                col_max = np.max(values)
                clean = clean_sentinels(raw, col_max)
                missing = np.isnan(clean)
                low = np.quantile(clean[~missing], self.clip_quantile)
                high = np.quantile(clean[~missing], 1 - self.clip_quantile)
                spec = {
                    "kind": "continuous",
                    "index": index,
                    "col_max": col_max,
                    "median": np.median(clean[~missing]),
                    "clip": (low, high),
                    "nan_indicator": np.sum(missing) >= min_count,
                }
                self.feature_names_.append(name)

            if spec["nan_indicator"]:
                self.feature_names_.append(f"{name}==NaN")
            self.specs_.append(spec)

        self.keep_ = np.ones(len(self.feature_names_), dtype=bool)
        z = self._build(x)
        self.keep_ = find_collinear_columns(z)

        kept = self.keep_[1:]
        mean = z[:, 1:].mean(axis=0, dtype=np.float64)[kept]
        std = z[:, 1:].std(axis=0, dtype=np.float64)[kept]
        self.mean_ = mean.astype(np.float32)
        std[std == 0] = 1
        self.std_ = std.astype(np.float32)

        kept_names = []
        for i in range(len(self.feature_names_)):
            if self.keep_[i]:
                kept_names.append(self.feature_names_[i])
        self.feature_names_ = kept_names
        return self

    def _build(self, x):
        """Build the feature matrix (not standardized yet) with only the
        columns marked in self.keep_."""
        out = np.empty((x.shape[0], np.sum(self.keep_)), dtype=np.float32)
        out[:, 0] = 1
        candidate = 1
        col = 1

        for spec in self.specs_:
            raw = np.array(x[:, spec["index"]], dtype=np.float64)

            new_features = []
            if spec["kind"] == "categorical":
                for level in spec["levels"]:
                    new_features.append(raw == level)
                missing = np.isnan(raw)
            else:
                clean = clean_sentinels(raw, spec["col_max"])
                missing = np.isnan(clean)
                clean[missing] = spec["median"]
                low, high = spec["clip"]
                new_features.append(np.clip(clean, low, high))
            if spec["nan_indicator"]:
                new_features.append(missing)

            for feature in new_features:
                if self.keep_[candidate]:
                    out[:, col] = feature
                    col += 1
                candidate += 1
        return out

    def transform(self, x):
        """Apply the learned preprocessing to x (train or test data).

        Returns a float32 matrix: bias column + standardized features.
        """
        out = self._build(x)
        out[:, 1:] -= self.mean_
        out[:, 1:] /= self.std_
        return out

    def fit_transform(self, x, columns):
        """Fit on x, then transform x."""
        self.fit(x, columns)
        return self.transform(x)
