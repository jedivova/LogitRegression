import pytest
from LogitRegression import LogitRegressor
from jit_trace import load_iris


@pytest.fixture
def data():
    data =  load_iris()
    return data

@pytest.mark.parametrize("max_iter, C",[(100,1),(110,10),(300,20),(300,0.1)])
def test_train_hyperparams(max_iter, C, data):
    X_train, y_train, X_test, y_test = data
    model = LogitRegressor(max_iter=max_iter, C=C)
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) > 0


