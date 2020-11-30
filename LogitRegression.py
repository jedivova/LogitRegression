import torch
import torch.nn as nn


import torch
import torch.nn as nn


class LogitRegressor(nn.Module):
    """
    Logistic Regression Classifier with l2 regularization

    Parameters
    ----------
    learning_rate : int or float, default=0.1
        The tuning parameter for the optimization algorithm (here, Gradient Descent)
        that determines the step size at each iteration while moving toward a minimum
        of the cost function.
    max_iter : int, default=1000
        Maximum number of iterations taken for the optimization algorithm to converge
    C : float, default=1
        Inverse of regularization strength; must be a positive float.
        Smaller values specify stronger regularization.
    tol : float, optional, default=1e-4
        Value indicating the weight change between epochs in which
        gradient descent should terminated.
    """

    def __init__(self, lr=0.1, max_iter=1000, C=1, tol=1e-4):
        super(LogitRegressor, self).__init__()
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.C = C

    def fit(self, X, Y, minibatch_len=10):
        """
        Fit the model withthe given training data.

        Parameters
        ---------
        X: torch.tensor of shape [n_samples, n_features]
            training data
        Y: torch.tensor of shape [n_samples,]
            training labels
        minibatch_len: int
            len of mini-batch

        Returns
        -------
        self:
            Fitted estimator.
        """

        self.w = torch.zeros(X.shape[1] + 1)  # adding 1 dim for bias
        _X = torch.cat((torch.ones((X.shape[0], 1)), X), dim=1)  # adding 1 dim for bias

        self.score_list = []  # write history of scores
        for _ in range(self.max_iter):
            ind = torch.randperm(_X.size(0))[:minibatch_len]  # each epoch shufle indexes
            x, y = _X[ind], Y[ind]  # take first 'minibatch_len' samples

            errors = self._sigmoid(x @ self.w) - y
            w_grad = self.lr * (x.T @ errors + 1 / self.C * self.w)

            if torch.abs(w_grad.mean()) > self.tol:
                self.w -= w_grad / x.shape[0]
            else:
                break

            self.score_list.append(self.score(x[:, 1:], y))
        return self


    def predict_proba(self, X):
        """
        Probability estimates for samples in X.

        Parameters
        ---------
        X: torch.tensor of shape [n_samples, n_features]
            Data

        Returns
        -------
        probabilities: torch.tensor of shape [n_samples]
            probabilities of each sample
        """

        return self._sigmoid((X @ self.w[1:]) + self.w[0])

    @torch.jit.export
    def predict(self, X):
        """
        Predict class labels for data X

        Parameters
        ---------
        X: torch.tensor of shape [n_samples, n_features]
            Data

        Returns
        -------
        labels: torch.tensor of shape [n_samples]
            predicted labels
        """

        return torch.round(self.predict_proba(X))

    def get_params(self, deep=True):
        return {'lr': self.lr,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'C': self.C}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @torch.jit.export
    def score(self, X, y):
        """
        score function of estimator

        Parameters
        ---------
        X: torch.tensor of shape [n_samples, n_features]
            Data
        y: torch.tensor of shape [n_samples,]
            labels
        Returns
        -------
        score: float
            accuracy of estimator on provided data
        """

        return (self.predict(X) == y).sum().item() / y.size(0)

    def _sigmoid(self, a):
        """
        Calculate sigmoid of a

        Parameters
        ---------
        a: torch.tensor
            Data

        Returns
        -------
        sigmoid: torch.tensor
        """

        return 1 / (1 + torch.exp(-a))