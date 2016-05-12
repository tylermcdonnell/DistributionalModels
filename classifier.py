from utils import *
from buildfeatures import SentimentFeatures, PatternFeatures
from buildfeatures import PartOfSpeechFeatures

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold

from abc import abstractmethod


class Classifier(object):
    def __init__(self, word_pairs_train, word_pairs_test, csv_train, csv_test):
        self.word_pairs_train = word_pairs_train
        self.word_pairs_test = word_pairs_test
        self.csv_train = csv_train
        self.csv_test = csv_test

        self.word_pairs = None
        self.data = None
        self.X = None
        self.y = None

        self.columns = ["sentiment_own", "sentiment_same_sent",
                        "sentiment_adj_sent", "pattern_either",
                        "pattern_neither", "pattern_from_to",
                        "dist_adverb_raw", "dist_adverb_lmi",
                        "dist_adverb_ppmi", "dist_noun_raw",
                        "dist_noun_lmi", "dist_noun_ppmi",
                        "dist_adjective_raw", "dist_adjective_lmi",
                        "dist_adjective_ppmi", "dist_verb_raw",
                        "dist_verb_lmi", "dist_verb_ppmi",
                        "dist_standard_raw", "dist_standard_lmi",
                        "dist_standard_ppmi"]

        self.model = None

    def __read_word_pairs(self, filename):
        word_pairs = {'original_tuple': [], 'stemmed_tuple': [], 'label': []}
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.split(' ')
                word_1 = tokens[1]
                word_2 = tokens[2]
                if word_2[-1] == '\n':
                    word_2 = word_2[0:-1]
                tup = sorted_tuple(word_1, word_2)
                word_pairs['original_tuple'].append(tup)

                word_1 = lemmatize_and_stem(word_1)
                word_2 = lemmatize_and_stem(word_2)
                # word_1 = stem_and_lemmatize(word_1)
                # word_2 = stem_and_lemmatize(word_2)
                tup = sorted_tuple(word_1, word_2)
                word_pairs['stemmed_tuple'].append(tup)

                word_pairs['label'].append(LABEL[tokens[0]])
        return word_pairs

    def __export_to_csv(self, csv_file):
        self.data.to_csv(csv_file, sep='\t', index=False)

    def __load_from_csv(self, csv_file):
        self.data = pd.read_csv(csv_file, sep='\t')

    def __extract_xy(self, data):
        y = data['label'].as_matrix().reshape(-1, 1).ravel()
        X = data.ix[:, self.columns].values
        return X, y

    def __load_data(self, csv_file, mode):
        if mode == BUILD_FEATURES:
            self.__build_feature_matrix(csv_file)
        elif mode == LOAD_FROM_CSV:
            self.__load_from_csv(csv_file)
        else:
            ValueError('Invalid mode specified.')

    def __build_feature_matrix(self, csv_file):
        self.data = pd.DataFrame(self.word_pairs)
        self.__add_features()
        self.__export_to_csv(csv_file)

    def __add_features(self):
        self.__sentiment_features()
        print("Sentiment Features loaded")
        self.__pattern_features()
        print("Pattern Features loaded")
        self.__distributional_features()
        print("Distributional Features loaded")

    def __sentiment_features(self):
        sf = SentimentFeatures(
            filename="./feature-dump/sentiment",
            target_words_filename="./WordLists/target_words.txt")
        tuples = self.word_pairs['stemmed_tuple']
        sentiment_feats = sf.sentiment_features(tuples)

        score_own = []
        score_same_sent = []
        score_adj_sent = []
        for tup in tuples:
            score_own.append(sentiment_feats[tup][OWN])
            score_same_sent.append(sentiment_feats[tup][SAME_SENT])
            score_adj_sent.append(sentiment_feats[tup][ADJ_SENT])
        self.data['sentiment_own'] = score_own
        self.data['sentiment_same_sent'] = score_same_sent
        self.data['sentiment_adj_sent'] = score_adj_sent

    def __pattern_helper(self, pf_object):
        tuples = self.word_pairs['stemmed_tuple']
        pattern_feats = pf_object.pattern_features(tuples)

        pattern_count = []
        for tup in tuples:
            if tup in pattern_feats:
                pattern_count.append(float(pattern_feats[tup]))
            else:
                pattern_count.append(0.0)
        return pattern_count

    def __pattern_features(self):
        either_pf = PatternFeatures("./feature-dump/pattern_either")
        either_count = self.__pattern_helper(either_pf)
        self.data['pattern_either'] = either_count

        neither_pf = PatternFeatures("./feature-dump/pattern_neither")
        neither_count = self.__pattern_helper(neither_pf)
        self.data['pattern_neither'] = neither_count

        from_to_pf = PatternFeatures("./feature-dump/pattern_from_to")
        from_to_count = self.__pattern_helper(from_to_pf)
        self.data['pattern_from_to'] = from_to_count

    def __distributional_helper(self, dist_object):
        tuples = self.word_pairs['stemmed_tuple']
        dist_feats = dist_object.pos_features(tuples)
        similarity = []
        for tup in tuples:
            similarity.append(float(dist_feats[tup]))
        return similarity

    def __distributional_features(self):
        pos_tags = ["standard", "adverb", "noun", "adjective", "verb"]
        metrics = [RAW, LMI, PPMI]

        for pos_tag in pos_tags:
            for metric in metrics:
                posf = PartOfSpeechFeatures(
                    "./feature-dump/" + pos_tag, metric)
                similarity = self.__distributional_helper(posf)
                del posf
                self.data['dist_' + pos_tag + '_' + metric] = similarity
                print(pos_tag, ",", metric, "done")

    def _cross_validation(self, X, y):
        n_folds = 5
        accuracies = []
        pr_f1 = []

        skf = StratifiedKFold(y=self.y, n_folds=n_folds, random_state=6)
        for train_idx, test_idx in skf:
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            accuracies.append(calc_accuracy(y_pred, y_test))
            pr_f1.append(calc_prec_recall_f1(y_pred, y_test))
            # class_accuracies.append(calc_class_wise_accuracy(y_pred, y_test))

        print("\n\nCross Validation Scores")
        self.__print_stats(accuracies, pr_f1)

    def _evaluate(self):
        accuracies = []
        pr_f1 = []
        y_pred = self.model.predict(self.X)
        accuracies.append(calc_accuracy(y_pred, self.y))
        pr_f1.append(calc_prec_recall_f1(y_pred, self.y))
        print("\n\nEvaluation Score")
        self.__print_stats(accuracies, pr_f1)

    def __print_stats(self, accuracies, pr_f1):
        print("Accuracies: ", accuracies)
        print("Accuracy mean = ", np.mean(accuracies))
        print("Accuracy sd = ", np.std(accuracies))
        print("---")
        print("Prec, Recall, F1, support", pr_f1)
        print("###############################")

    def training_data(self, mode):
        self.word_pairs = self.__read_word_pairs(self.word_pairs_train)
        self.__load_data(self.csv_train, mode)
        self.X, self.y = self.__extract_xy(self.data)

    def testing_data(self, mode):
        self.word_pairs = self.__read_word_pairs(self.word_pairs_test)
        self.__load_data(self.csv_test, mode)
        self.X, self.y = self.__extract_xy(self.data)

    @abstractmethod
    def grid_search(self):
        pass

    @abstractmethod
    def fit_model(self):
        pass


def main():
    c = Classifier(word_pairs_train="./WordLists/train.txt",
                   word_pairs_test="./WordLists/test.txt",
                   csv_train="feature_matrix_train.csv",
                   csv_test="feature_matrix_test.csv")
    c.training_data(mode=BUILD_FEATURES)
    c.testing_data(mode=BUILD_FEATURES)
    # c.training_data(mode=LOAD_FROM_CSV)
    # c.testing_data(mode=LOAD_FROM_CSV)

if __name__ == '__main__':
    main()
