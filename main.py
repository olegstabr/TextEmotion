# -*- coding: utf-8 -*-
import numpy as np
from enum import Enum
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn import metrics


#Объединение массивов положительных и отрицательных текстов в единый исходный массив данных - common_data
positive_path = "./data/twitter/positive.csv"
negative_path = "./data/twitter/negative.csv"
positive_data = read_csv(positive_path, ";")
negative_data = read_csv(negative_path, ";")
negative_data = np.concatenate((positive_data, negative_data), axis=0)
common_data = np.concatenate((positive_data, negative_data), axis=0)
# берём 4 и 5 столбцы - текст сообщения и тип твита
X = common_data[:, 3]
Y = common_data[:, 4]


class Classifier(Enum):
    MULTINOMIAL=1
    BERNOULLI=2
    GAUSSIAN=3


def classify(maxf, size, cl):
    # преобразование исходных текстов сообщений в вектор атрибутов
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=maxf)
    train_data_features = vectorizer.fit_transform(X)
    train_data_features = train_data_features.toarray()

    # разделение на тестовую и обучающую выборки
    X_train, X_test, Y_train, Y_test = train_test_split(train_data_features, Y, test_size=size, random_state=0)
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)

    # выбор классификатора
    if cl == Classifier.MULTINOMIAL: clf = MultinomialNB()
    if cl == Classifier.BERNOULLI: clf = BernoulliNB()
    if cl == Classifier.GAUSSIAN: clf = GaussianNB()

    # построение модели, прогнозирование значений, вывод оценки качества
    clf.fit(X_train, Y_train)
    res = clf.predict(X_test)
    print metrics.accuracy_score(Y_test, res)


classify(500, 0.3, Classifier.BERNOULLI)
# classify(700, 0.3, Classifier.BERNOULLI)
# classify(900, 0.3, Classifier.BERNOULLI)
# classify(500, 0.3, Classifier.MULTINOMIAL)
# classify(700, 0.3, Classifier.MULTINOMIAL)
# classify(900, 0.3, Classifier.MULTINOMIAL)
