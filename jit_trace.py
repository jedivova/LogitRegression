import torch
from LogitRegression import LogitRegressor

def load_iris():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    iris = datasets.load_iris()
    ind = iris.target != 2

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data[ind],
        iris.target[ind],
        test_size=0.2,
        random_state=42)

    def preprocess(A):
        return torch.from_numpy(A).float()

    return [preprocess(X_train),
            preprocess(y_train),
            preprocess(X_test),
            preprocess(y_test)]


def main():
    X_train, y_train, X_test, y_test = load_iris()

    model = LogitRegressor()
    model.fit(X_train, y_train)
    print('accuracy on test data =',model.score(X_test, y_test))

    scripted_module = torch.jit.script(model)
    torch.jit.save(scripted_module, 'scriptmodule.pt')
    loaded_model = torch.jit.load('scriptmodule.pt')
    print('traced model accuracy =', loaded_model.score(X_test, y_test))


if __name__ == "__main__":
    main()