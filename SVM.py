import numpy as np

from sklearn.base import BaseEstimator

def add_bias(X):
    return np.hstack((X, np.ones(shape=(X.shape[0], 1))))

class SVM(BaseEstimator):
    def __init__(self, lr, reg, epoch=64):
        BaseEstimator.__init__(self)

        self.reg = reg
        self.lr = lr
        self.epoch = epoch

    def fit(self, X, y):
        X = add_bias(X)
        self.W = np.zeros(shape=(X.shape[1], 1))
        examples = np.hstack((X, y.reshape(y.shape[0], -1)))

        champ = self.W
        for t in range(self.epoch):
            lr_t = self.lr/(1 + t)

            # shuffle training set
            np.random.shuffle(examples)
            for example in examples:
                example = example.reshape((-1, 1))
                x, label = example[:-1], float(example[-1])
                if np.all(label * np.dot(self.W.T, x) <= 1):
                    self.W = (1 - lr_t) * self.W + lr_t * self.reg * label * x
                else:
                    self.W = (1 - lr_t) * self.W

            champ_score = np.mean(np.equal(np.sign(np.dot(X, champ)), y.reshape(-1, 1)))
            new_score = np.mean(np.equal(np.sign(np.dot(X, self.W)), y.reshape(-1, 1)))
            champ = champ if champ_score >= new_score else self.W

        self.W = champ

    def predict(self, X):
        X = add_bias(X)
        return np.sign(np.dot(X, self.W))

    def score(self, X, y):
        return np.mean(np.equal(self.predict(X), y.reshape(-1, 1)))

    def get_params(self, deep=True):
        return {'reg': self.reg,
                'lr': self.lr,
                'epoch': self.epoch}

    def set_params(self, **params):
        # fixme This is going to have to change
        return self
