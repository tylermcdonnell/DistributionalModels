from classifier import Classifier
from utils import *

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticRegressionClf(Classifier):
    def __init__(self, word_pairs_train, word_pairs_test, csv_train, csv_test):
        super(LogisticRegressionClf, self).__init__(word_pairs_train,
                                                    word_pairs_test,
                                                    csv_train,
                                                    csv_test)

    def grid_search(self):
        self.training_data(LOAD_FROM_CSV)
        n_folds = 5
        param_grid = {'C': np.linspace(0.01, 0.2, 20)}

        model = LogisticRegression(penalty='l2',
                                   solver='lbfgs',
                                   n_jobs=-1,
                                   random_state=6)

        gs = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='f1_micro',
                          n_jobs=1,
                          cv=n_folds)

        gs.fit(self.X, self.y)

        print(gs.grid_scores_)
        print(gs.best_score_)
        print(gs.best_params_)

    def fit_model(self):
        self.training_data(LOAD_FROM_CSV)
        self.model = LogisticRegression(C=0.013,
                                        penalty='l2',
                                        solver='lbfgs',
                                        n_jobs=-1,
                                        random_state=6)
        self._cross_validation(self.X, self.y)

    def test_model(self):
        self.testing_data(LOAD_FROM_CSV)
        self._evaluate()


def main():
    lr = LogisticRegressionClf(word_pairs_train="./WordLists/train.txt",
                               word_pairs_test="./WordLists/test.txt",
                               csv_train="feature_matrix_train.csv",
                               csv_test="feature_matrix_test.csv")
    # lr.grid_search()
    lr.fit_model()
    lr.test_model()

if __name__ == '__main__':
    main()
