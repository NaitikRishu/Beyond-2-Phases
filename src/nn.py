"""Minimal NumPy re-implementation of the paper's DNN: tanh hidden layers, dropout, Adam, cross-entropy.
(TensorFlow 1.x DNNClassifier was used originally; PyTorch/TF were unavailable in our environment.)"""
import numpy as np
from sklearn.metrics import roc_auc_score


class MLP:
    def __init__(self, n_in, hidden=(5, 5), dropout=0.2, lr=1e-3, l2=0.0, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [rng.normal(0, np.sqrt(1.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.dropout, self.lr, self.l2, self.rng = dropout, lr, l2, rng
        self.m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.t = 0

    def _forward(self, X, train):
        acts, masks = [X], []
        h = X
        for i in range(len(self.W) - 1):
            h = np.tanh(h @ self.W[i] + self.b[i])
            if train and self.dropout > 0:
                mk = (self.rng.random(h.shape) > self.dropout) / (1 - self.dropout)
                h = h * mk
            else:
                mk = None
            masks.append(mk); acts.append(h)
        z = h @ self.W[-1] + self.b[-1]
        return 1 / (1 + np.exp(-np.clip(z, -30, 30))), acts, masks

    def predict_proba(self, X):
        out = np.empty(len(X))
        for s in range(0, len(X), 100000):
            out[s:s + 100000] = self._forward(X[s:s + 100000], False)[0][:, 0]
        return out

    def _step(self, X, y):
        p, acts, masks = self._forward(X, True)
        n = len(X)
        delta = (p[:, 0] - y)[:, None] / n           # dL/dz for sigmoid + BCE
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            grads_W[i] = acts[i].T @ delta + self.l2 * self.W[i]
            grads_b[i] = delta.sum(0)
            if i > 0:
                dh = delta @ self.W[i].T
                if masks[i - 1] is not None: dh = dh * masks[i - 1]
                # acts[i] is post-dropout; recover tanh value for derivative
                hval = acts[i] if masks[i - 1] is None else np.divide(acts[i], masks[i - 1], out=np.zeros_like(acts[i]), where=masks[i - 1] != 0)
                delta = dh * (1 - hval ** 2)
        self.t += 1
        params = self.W + self.b; grads = grads_W + grads_b
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, (prm, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            prm -= self.lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, y, Xval, yval, epochs=40, batch=256, patience=4):
        best, best_state, bad = -1, None, 0
        for ep in range(epochs):
            idx = self.rng.permutation(len(X))
            for s in range(0, len(X), batch):
                j = idx[s:s + batch]; self._step(X[j], y[j])
            auc = roc_auc_score(yval, self.predict_proba(Xval))
            if auc > best + 1e-4:
                best, bad = auc, 0
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                bad += 1
                if bad >= patience: break
        self.W, self.b = best_state
        self.best_val_auc, self.epochs_run = best, ep + 1
        return self
