from classifier import Classifier
from utils import *

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np


class SvmClf(Classifier):
    def __init__(self, word_pairs_train, word_pairs_test, csv_train, csv_test):
        super(SvmClf, self).__init__(word_pairs_train,
                                     word_pairs_test,
                                     csv_train,
                                     csv_test)

    def grid_search(self):
        self.training_data(LOAD_FROM_CSV)
        n_folds = 5
        param_grid = {'C': np.linspace(0.5, 0.8, 20)}

        model = SVC(random_state=6)

        gs = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          scoring='accuracy',
                          n_jobs=1,
                          cv=n_folds)

        gs.fit(self.X, self.y)

        print(gs.grid_scores_)
        print(gs.best_score_)
        print(gs.best_params_)

    def fit_model(self):
        '''
        Accuracy mean =  0.592375175991
        Accuracy sd =  0.0269336673676
        Class-wise accuracy mean=
        mean= 0.0 sd= 0.0
        mean= 0.849738230687 sd= 0.0336791809521
        mean= 0.516012084592 sd= 0.0902792637855
        '''
        self.training_data(LOAD_FROM_CSV)
        self.model = SVC(C=0.721,
                         gamma='auto',
                         kernel='rbf',
                         random_state=6)
        self._cross_validation(self.X, self.y)

    def test_model(self):
        self.testing_data(LOAD_FROM_CSV)
        self._evaluate()


def main():
    svm = SvmClf(word_pairs_train="./WordLists/train.txt",
                 word_pairs_test="./WordLists/test.txt",
                 csv_train="feature_matrix_train.csv",
                 csv_test="feature_matrix_test.csv")
    # svm.grid_search()
    svm.fit_model()
    svm.test_model()

if __name__ == '__main__':
    main()
