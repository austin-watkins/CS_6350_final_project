import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import SGDClassifier
from SVM import SVM
from itertools import product
from statistics import mean
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def get_data_numpy(file):
    train_data = pd.read_csv(f'data/recipe/train/train_data_{file}.csv')
    y_train = train_data['label']
    y_train = y_train.to_numpy()
    X_train = train_data.drop('label', axis=1)
    X_train = X_train.to_numpy()

    test_data = pd.read_csv(f'data/recipe/train/train_data_{file}.csv')
    y_test = test_data['label']
    y_test = y_test.to_numpy()
    X_test = test_data.drop('label', axis=1)
    X_test = X_test.to_numpy()

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    for recipe in range(1, 5):
        (X_train_data, y_train_data), (X_test_data, y_test_data) = get_data_numpy(file=recipe)
        loo = LeaveOneOut()

        lrs = []
        regs = []
        for i in range(-4, 1):
            lrs.append(10**i)
            regs.append(10**i)

        classes = np.unique(y_train_data)
        class_weights = compute_class_weight('balanced', classes, y_train_data)
        class_weights = {c: w for (c, w) in zip(classes, class_weights)}

        # this is for linearSVM
        # this is for my svm

        results = []

        # THIS IS WHERE I SELECT MODELS
        model_type = 'SGDClassifier'
        # model = 'SVM'

        file = open(f'results/{model_type}_{recipe}.txt', 'w')
        file.write(f'MODEL: {model_type}\n')

        if model_type == 'SGDClassifier':
            model = SGDClassifier
            args = [{'eta0': lr, 'learning_rate': 'invscaling', 'alpha': reg,'class_weight': class_weights} for (lr, reg) in product(lrs, regs)]
        elif model_type == 'SVM':
            model = SVM
            args = [{'lr': lr, 'reg': reg, 'epoch': 50} for (lr, reg) in product(lrs, regs)]
        else:
            raise ValueError('model not found')


        for a in args:
            svc = model(**a)
            cv_results = cross_val_score(svc, X_train_data, y_train_data, cv=loo.split(X_train_data, y_train_data), n_jobs=-1)
            average_score = mean(cv_results)
            results.append((average_score, a))
            file.write(f'argument: \t\t{a}\n')
            file.write(f'average score: \t{average_score:.4}\n\n')
            break

        file.write('FINISHED TRAINING\n')
        file.write('Arguments sorted by accuracy:\n')
        results = sorted(results, key=lambda x: x[0], reverse=True)
        for r in results:
            file.write(str(r) + '\n')
        file.write('\n')

        best_args = results[0][1]
        file.write(f'best_args: \t{best_args}\n')
        file.write(f'training with \t{best_args}\n')
        svc = model(**best_args)
        svc.fit(X_train_data, y_train_data)
        file.write('\n')
        file.write('Accuracy:\n')
        file.write(f'training accuracy is {svc.score(X_train_data, y_train_data)}\n')
        file.write(f'test accuracy is {svc.score(X_test_data, y_test_data)}\n')

        file.write('\n')
        file.write('F1:\n')
        file.write(f'training F1 is {f1_score(y_train_data, svc.predict(X_train_data))}\n')
        file.write(f'test F1 is {f1_score(y_test_data, svc.predict(y_test_data))}\n')
