import numpy as np
from utils import entropy, visualize, visualize_1
from random import sample


class Node:
    def __init__(self, depth, gain, prev_E, X, Y):
        self.depth = depth
        self.gain = gain
        self.prev_E = prev_E
        self.p = None
        self.pred = None
        unique, counts = np.unique(Y, return_counts=True)
        perc = counts/np.sum(counts)
        # se sono arrivato alla massima profondità oppure se sono arrivato ad avere meno di 10 elementi
        # oppure se una classe è dominante sulle altre
        if depth == 0 or X.shape[0] < 10 or np.max(perc) > 0.9:
            self.depth = 0
            self.p = counts / Y.size
            try:
                self.pred = unique[np.argmax(self.p)]
            except:
                self.pred = 0
        else:
            self.thresh = None
            self.left = None
            self.right = None
            self.dim = None
            self.fit(X, Y)

    def fit(self, X, Y):
        self.dim = np.random.choice(range(X.shape[1]))
        x = np.array(X[:, self.dim])
        thresh_tests = 10
        unit = (np.max(x) - np.min(x)) / thresh_tests
        best_E = - 10000000

        for i in range(1, 10):
            thresh = np.min(x) + i * unit
            left_size = np.count_nonzero(x < thresh)
            right_size = np.count_nonzero(x >= thresh)
            # ratio = np.min([left_size, right_size]) / np.max([left_size, right_size])
            delta_E = - (left_size/x.size) * entropy(Y[x < thresh]) - (right_size/x.size) * entropy(Y[x >= thresh])

            if delta_E > best_E:
                best_E = delta_E
                self.thresh = thresh

        # visualize(X[X[:, self.dim] < self.thresh], X[X[:, self.dim] >= self.thresh], 'red', 'blue')
        # visualize_1(X[x < self.thresh], Y[x < self.thresh])
        # visualize_1(X[x >= self.thresh], Y[x >= self.thresh])
        if np.abs((np.max([best_E, self.prev_E]) / np.min([best_E, self.prev_E]))) < self.gain:
            self.left = Node(self.depth-1, self.gain, best_E, X[x < self.thresh], Y[x < self.thresh])
            self.right = Node(self.depth-1, self.gain, best_E, X[x >= self.thresh], Y[x >= self.thresh])
        else:
            _, counts = np.unique(Y, return_counts=True)
            self.p = counts / Y.size
            self.pred = np.argmax(self.p)
            self.depth = 0

    # predizione di un singolo elemento, X è un array, non matrice
    def predict(self, X):
        if self.depth == 0:
            return self.pred
        if X[self.dim] < self.thresh:
            return self.left.predict(X)
        else:
            return self.right.predict(X)

    def get_probabilities(self):
        return self.p


class DT:
    def __init__(self, max_depth, gain):
        self.max_depth = max_depth
        self.gain = gain
        self.node = None

    def fit(self, X, Y):
        self.node = Node(self.max_depth, self.gain, -10000, X, Y)

    def predict(self, X):
        Y_pred = []
        for x in X:
            Y_pred.append(self.node.predict(x))
        return Y_pred


class Forest:
    def __init__(self, n=50, gain=1.5, train_perc=0.8, max_depth=10):
        self.gain = gain
        self.train_perc = train_perc
        self.max_depth = max_depth
        self.n = n
        self.forest = []

    def fit(self, X, Y):
        dts = []
        targets = []
        idx = [sample(range(X.shape[0]), round(X.shape[0] * self.train_perc)) for _ in range(self.n)]
        for i in range(self.n):
            dts.append([X[j] for j in idx[i]])
            targets.append([Y[j] for j in idx[i]])

        for i in range(self.n):
            self.forest.append(DT(self.max_depth, self.gain))
            self.forest[i].fit(np.asarray(dts[i]), np.asarray(targets[i]))

    def predict(self, X):
        preds = np.zeros((self.n, X.shape[0]))
        P = []
        for i in range(self.n):
            preds[i] = self.forest[i].predict(X)

        for i in range(X.shape[0]):
            arr = preds[:, i]
            unique, counts = np.unique(arr, return_counts=True)
            P.append(unique[np.argmax(counts)])
        return [P, preds]

    def score(self, P, Y):
        return float(np.sum(P == Y)) / Y.size

    def score_mean(self, preds, Y):
        tree_score = []
        for i in range(self.n):
            tree_score.append(float(np.sum(preds[i, :] == Y)) / Y.size)
        return np.mean(tree_score)
