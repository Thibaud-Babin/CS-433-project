import numpy as np

N_BINS = 64
NAN_BIN = N_BINS - 1


def fit_bin_edges(x, columns):
    edges = []
    for j in columns:
        col = np.array(x[:, j], dtype=np.float64)
        col = col[~np.isnan(col)]
        if len(col) == 0:
            edges.append(np.zeros(0))
            continue
        values = np.unique(col)
        if len(values) <= N_BINS - 1:
            e = (values[:-1] + values[1:]) / 2
        else:
            q = np.linspace(0, 1, N_BINS)[1:-1]
            e = np.unique(np.quantile(col, q))
        edges.append(e)
    return edges


def apply_bins(x, columns, edges):
    out = np.empty((x.shape[0], len(columns)), dtype=np.uint8)
    for k in range(len(columns)):
        col = np.array(x[:, columns[k]], dtype=np.float64)
        b = np.searchsorted(edges[k], col, side="right")
        b[np.isnan(col)] = NAN_BIN
        out[:, k] = b
    return out


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss(y, raw):
    return np.mean(np.logaddexp(0, raw) - y * raw)


def build_histograms(xt, features, rows, g, h):
    G = np.zeros((len(features), N_BINS))
    H = np.zeros((len(features), N_BINS))
    g_rows = g[rows]
    h_rows = h[rows]
    for k in range(len(features)):
        bins = xt[features[k], rows]
        G[k] = np.bincount(bins, weights=g_rows, minlength=N_BINS)
        H[k] = np.bincount(bins, weights=h_rows, minlength=N_BINS)
    return G, H


def best_split(G, H, lam, min_child_hess):
    g_total = G[0].sum()
    h_total = H[0].sum()
    parent_score = g_total**2 / (h_total + lam)
    g_left = np.cumsum(G[:, :-1], axis=1)
    h_left = np.cumsum(H[:, :-1], axis=1)
    g_nan = G[:, -1:]
    h_nan = H[:, -1:]

    best = (0.0, -1, -1, False)
    for nan_left in [False, True]:
        GL = g_left
        HL = h_left
        if nan_left:
            GL = g_left + g_nan
            HL = h_left + h_nan
        GR = g_total - GL
        HR = h_total - HL
        gain = GL**2 / (HL + lam) + GR**2 / (HR + lam) - parent_score
        gain[(HL < min_child_hess) | (HR < min_child_hess)] = -np.inf
        f, t = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[f, t] > best[0]:
            best = (gain[f, t], f, t, nan_left)
    return best


class Tree:

    def __init__(self):
        self.feature = []
        self.threshold = []
        self.nan_left = []
        self.left = []
        self.right = []
        self.value = []
        self.depth = 0

    def add_node(self, value):
        self.feature.append(-1)
        self.threshold.append(0)
        self.nan_left.append(False)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(value)
        return len(self.value) - 1

    def to_arrays(self):
        self.feature_arr = np.array(self.feature)
        self.threshold_arr = np.array(self.threshold)
        self.nan_left_arr = np.array(self.nan_left)
        self.left_arr = np.array(self.left)
        self.right_arr = np.array(self.right)
        self.value_arr = np.array(self.value)

    def predict(self, xb):
        node = np.zeros(xb.shape[0], dtype=int)
        rows = np.arange(xb.shape[0])
        for step in range(self.depth):
            f = self.feature_arr[node]
            is_leaf = f < 0
            bins = xb[rows, np.maximum(f, 0)]
            go_left = bins <= self.threshold_arr[node]
            go_left = go_left | ((bins == NAN_BIN) & self.nan_left_arr[node])
            child = np.where(go_left, self.left_arr[node], self.right_arr[node])
            node = np.where(is_leaf, node, child)
        return self.value_arr[node]


class GBDT:

    def __init__(
        self,
        n_trees=1500,
        learning_rate=0.1,
        max_depth=3,
        lam=1.0,
        min_child_hess=5.0,
        subsample=0.5,
        colsample=0.5,
        early_stopping=50,
        seed=0,
        verbose=0,
    ):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lam = lam
        self.min_child_hess = min_child_hess
        self.subsample = subsample
        self.colsample = colsample
        self.early_stopping = early_stopping
        self.seed = seed
        self.verbose = verbose

    def grow_tree(self, xt, g, h, rows, features):
        tree = Tree()
        G, H = build_histograms(xt, features, rows, g, h)
        self.split_node(tree, xt, g, h, rows, features, G, H, 0)
        tree.depth = self.max_depth
        tree.to_arrays()
        return tree

    def split_node(self, tree, xt, g, h, rows, features, G, H, depth):
        value = -self.learning_rate * G[0].sum() / (H[0].sum() + self.lam)
        node = tree.add_node(value)
        if depth == self.max_depth:
            return node
        gain, f, t, nan_left = best_split(G, H, self.lam, self.min_child_hess)
        if gain <= 0:
            return node

        bins = xt[features[f], rows]
        go_left = (bins <= t) | ((bins == NAN_BIN) & nan_left)
        left_rows = rows[go_left]
        right_rows = rows[~go_left]
        if len(left_rows) <= len(right_rows):
            GL, HL = build_histograms(xt, features, left_rows, g, h)
            GR, HR = G - GL, H - HL
        else:
            GR, HR = build_histograms(xt, features, right_rows, g, h)
            GL, HL = G - GR, H - HR

        tree.feature[node] = features[f]
        tree.threshold[node] = t
        tree.nan_left[node] = nan_left
        left = self.split_node(tree, xt, g, h, left_rows, features, GL, HL, depth + 1)
        right = self.split_node(tree, xt, g, h, right_rows, features, GR, HR, depth + 1)
        tree.left[node] = left
        tree.right[node] = right
        return node

    def fit(self, xb, y, xb_val, y_val):
        rng = np.random.default_rng(self.seed)
        n = xb.shape[0]
        n_features = xb.shape[1]
        p0 = np.mean(y)
        self.base = np.log(p0 / (1 - p0))
        raw = np.full(n, self.base)
        raw_val = np.full(len(y_val), self.base)

        xt = np.ascontiguousarray(xb.T)
        n_rows = int(round(self.subsample * n))
        n_cols = max(1, int(round(self.colsample * n_features)))

        self.trees = []
        self.val_loss = []
        best_loss = np.inf
        best_n_trees = 0
        for i in range(self.n_trees):
            p = sigmoid(raw)
            g = p - y
            h = p * (1 - p)
            rows = np.sort(rng.choice(n, n_rows, replace=False))
            features = np.sort(rng.choice(n_features, n_cols, replace=False))
            tree = self.grow_tree(xt, g, h, rows, features)
            self.trees.append(tree)
            raw += tree.predict(xb)
            raw_val += tree.predict(xb_val)

            loss = log_loss(y_val, raw_val)
            self.val_loss.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_n_trees = i + 1
            elif i + 1 - best_n_trees >= self.early_stopping:
                break
            if self.verbose and (i + 1) % self.verbose == 0:
                print(f"  tree {i + 1}: validation loss {loss:.5f}", flush=True)

        self.trees = self.trees[:best_n_trees]
        return self

    def decision_function(self, xb):
        raw = np.full(xb.shape[0], self.base)
        for tree in self.trees:
            raw += tree.predict(xb)
        return raw

    def split_counts(self, n_features):
        counts = np.zeros(n_features)
        for tree in self.trees:
            for f in tree.feature:
                if f >= 0:
                    counts[f] += 1
        return counts
