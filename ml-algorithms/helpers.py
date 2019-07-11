import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import datetime

from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, hinge_loss, f1_score
from sklearn.model_selection import StratifiedKFold

from statistics import mean

plot_labels = {
    "acc": "Accuracy",
    "loss": "Loss",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score"
}

def plot_to_file(histories, history_names, plot_types, save_file=None, algorithm="undefined"):
    for plot_type in plot_types:
        try:
            plot_label = plot_labels[plot_type]
        except KeyError:
            print("Please use a valid plot type (acc, loss)")
            return

        if not save_file:
            print("Please add a save file name")
            return
        else:
            legends = []
            for name in history_names:
                legends.append("{}_Train".format(name))
                legends.append("{}_Test".format(name))

            for index, history in enumerate(histories):
                plt.plot(history.history[plot_type], color='red')
                plt.plot(history.history['val_{}'.format(plot_type)], color='blue')

            plt.title('Model {}'.format(plot_label))
            plt.ylabel(plot_label)
            plt.xlabel('Epoch')
            plt.legend(['training', 'validation'], loc='upper left')
            plt.savefig("plots/{}_{}_{}.ps".format(plot_type, algorithm, save_file))
            plt.clf()

def pretty_print_scores(scores, names, churn_numbers, total_numbers, y_preds):
    losses = list(map(lambda x: x[0], scores))
    accs = list(map(lambda x: x[1], scores))
    precisions = list(map(lambda x: x[2], scores))
    recalls = list(map(lambda x: x[3], scores))
    f1_scores = list(map(lambda x: x[4], scores))

    for index, name in enumerate(names):
        pred_pos = list(filter(lambda x: x==1, y_preds[index]))
        print("--- {}".format(name))
        print("\tLoss:\t{}".format(losses[index]))
        print("\tAcc: \t{}".format(accs[index]))
        print("\tPrecision: \t{}".format(precisions[index]))
        print("\tRecall: \t{}".format(recalls[index]))
        print("\tF1-score: \t{}".format(f1_scores[index]))
        print("\tSamples: {}/{}".format(churn_numbers[index], total_numbers[index]))
        print("\tPredicted positives: {}/{}".format(len(pred_pos), len(y_preds[index])))

    print("Loss avg: \t {}".format(mean(losses)))
    print("Acc avg: \t {}".format(mean(accs)))
    print("Precision avg: \t {}".format(mean(precisions)))
    print("Recall avg: \t {}".format(mean(recalls)))
    print("F1-score avg: \t {}".format(mean(f1_scores)))

        

def log_to_csv(network_type, score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model, nb_filters=0, kernel_size=0):
    history_values = list(map(lambda x: ",".join(map(str, x)), history.history.values()))
    loss = score[0]
    acc = score[1]
    filters_as_str = ",".join(map(str, filters))
    row = [datetime.datetime.now(), network_type, table_folder, filters_as_str, input_dim, batch_size, time_from, time_to, model, nb_filters, kernel_size, loss, acc] + history_values

    with open('network_runs.csv', 'a', newline="\n") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(row)

def print_feature_importances(feature_importances, feature_names, all=False):
    feature_weights = {}
    
    for i in range(0, len(feature_importances)):
        try:
            feature_weights[feature_names[i%len(feature_names)]] = feature_weights[feature_names[i%len(feature_names)]] + [feature_importances[i]]
        except KeyError:
            feature_weights[feature_names[i%len(feature_names)]] = [feature_importances[i]]
        
    feature_weight_sums = {k: sum(v) for k, v in feature_weights.items()}
    j = 0
    for i in sorted(feature_weight_sums, key=feature_weight_sums.get, reverse=True):
        j = j + 1
        print("{}: {}".format(i, feature_weight_sums[i]))
        if j == 5 and not all:
            break

def plot_svm(train_sizes, train_scores, validation_scores, save_file):
    for i in range(0, len(train_sizes)):
        train_scores_means = []
        validation_scores_means = []
        train_size = train_sizes[i]
        train_score = train_scores[i]
        validation_score = validation_scores[i]
        
        for j in range(0, len(train_size)):
            train_scores_means.append(train_score[j].mean())
            validation_scores_means.append(validation_score[j].mean())
        
        plt.plot(train_size, np.array(train_scores_means), color='red', label='training')
        plt.plot(train_size, np.array(validation_scores_means), color='blue', label='validation')

    plt.ylabel('Precision score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Model RF', fontsize = 18, y = 1.03)
    plt.legend(["training", "validation"])
    plt.ylim(0, 1.5)
    plt.savefig("plots/learning_rf_{}.ps".format(save_file))

def pretty_print_scores_svm(scores, history_names, churn_numbers, total_numbers, y_preds, y_tests, feature_importances, feature_names):
    accuracies = list(map(lambda x: x[0], scores))
    precisions = list(map(lambda x: x[1], scores))
    recalls = list(map(lambda x: x[2], scores))
    losses = list(map(lambda x: x[3], scores))
    f1_scores = list(map(lambda x: x[4], scores))

    for i in range(0, len(history_names)):
        pred_pos = list(filter(lambda x: x==1, y_preds[i]))
        true_pos = list(filter(lambda x: x==1, y_tests[i]))
        print("--- {}".format(history_names[i]))
        print("\tAccuracy: {}".format(accuracies[i]))
        print("\tPrecision: {}".format(precisions[i]))
        print("\tRecall: {}".format(recalls[i]))
        print("\tHinge loss: {}".format(losses[i]))
        print("\tF1-score: \t{}".format(f1_scores[i]))
        print("\tSamples: {}/{}".format(churn_numbers[i], total_numbers[i]))
        print("\tPredicted positives: {}/{}".format(len(pred_pos), len(true_pos)))
    
    print("Loss avg: \t {}".format(mean(losses)))
    print("Acc avg: \t {}".format(mean(accuracies)))
    print("Precision avg: \t {}".format(mean(precisions)))
    print("Recall avg: \t {}".format(mean(recalls)))
    print("F1-score avg: \t {}".format(mean(f1_scores)))

    # for i in feature_importances:
    #     try:
    #         total_feature_importances = total_feature_importances + i
    #     except NameError:
    #         total_feature_importances = np.array([0]*i)
        
    # total_feature_importances = total_feature_importances / len(feature_importances)
    # print_feature_importances(total_feature_importances, feature_names[0], all=True)

def get_score(y, y_pred, scoring=None):
    score_indices = {
        'accuracy': 0,
        'precision': 1,
        'recall': 2,
        'hinge_loss': 3,
        'f1': 4
    }

    train_score = [accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), hinge_loss(y, y_pred), f1_score(y, y_pred)]
    if scoring:
        score_index = score_indices[scoring]
        train_score = np.array(train_score[score_index])
    return train_score

def training_curve(clf, X_train, y_train, X_test, y_test, train_sizes, shuffle, scoring, train_last=False):
    train_scores = []
    validation_scores = []
    if train_last:
        train_sizes = train_sizes + [len(y_train)]
    for size in train_sizes:
        batch_X_train, batch_y_train = resample(X_train, y_train, n_samples=size)
        clf.fit(batch_X_train, batch_y_train)
        
        train_y_pred = clf.predict(X_train)
        train_scores.append(get_score(y_train, train_y_pred, scoring))

        val_y_pred = clf.predict(X_test)
        validation_scores.append(get_score(y_test, val_y_pred, scoring))

    return np.array(train_sizes), np.array(train_scores), np.array(validation_scores)

gammas = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
c_ranges = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
kernels = ['linear', 'rbf']
hyperparams = {
    'C': c_ranges,
    'gamma': gammas,
    'kernel': kernels
}

def k_fold(X_train, y_train, folds):
    folded_X_train = []
    folded_y_train = []
    fold_length = len(y_train)/folds
    for i in range(0, folds-1):
        fold_index = int(i*fold_length)
        next_fold_index = int(i+1*fold_length)
        folded_X_train.append(X_train[fold_index:next_fold_index])
        folded_y_train.append(y_train[fold_index:next_fold_index])
    return folded_X_train, folded_y_train

def grid_search_svm(clf, c_ranges, gammas, kernels, X_train, y_train, X_test, y_test):
    optimals_list = []
    total_iterations = len(c_ranges)*len(gammas)*len(kernels)
    iteration=1
    fold_amount = 5
    skf = StratifiedKFold(n_splits=fold_amount)
    
    fold_train = k_fold(X_train, y_train, fold_amount)
    for c_value in c_ranges:
        for gamma_value in gammas:
            for kernel_value in kernels:
                print("Iteration {}/{} - c: {}, gamma: {}, kernel: {}".format(iteration, total_iterations, c_value, gamma_value, kernel_value))
                params = {
                    'C': c_value,
                    'gamma': gamma_value,
                    'kernel': kernel_value
                }
                clf.set_params(**params)

                f = lambda x: x/fold_amount
                y_scores = np.array([0]*5)
                y_preds = []
                pred_pos_list = []

                for train_index, test_index in skf.split(X_train, y_train):
                    folded_X_train = np.array(X_train)[test_index]
                    folded_y_train = np.array(y_train)[test_index]
                    clf.fit(folded_X_train, folded_y_train)
                    y_pred = clf.predict(X_test)
                    y_scores = y_scores + np.array(get_score(y_test, y_pred))
                    pred_pos = len(list(filter(lambda x: x==1, y_pred)))
                    y_preds.append(y_pred)
                    pred_pos_list.append(pred_pos)

                print(np.array(pred_pos).mean())
                if np.array(pred_pos).mean() >= 7:
                    optimals_list.append([f(y_scores), c_value, gamma_value, kernel_value])

                iteration = iteration + 1

    accuracy_optimals = list(map(lambda x: [x[0][0]] + x[1:], optimals_list))
    precision_optimals = list(map(lambda x: [x[0][1]] + x[1:], optimals_list))
    recall_optimals = list(map(lambda x: [x[0][2]] + x[1:], optimals_list))
    hinge_loss_optimals = list(map(lambda x: [x[0][3]] + x[1:], optimals_list))
    f1_optimals = list(map(lambda x: [x[0][4]] + x[1:], optimals_list))

    best_accuracy = max(accuracy_optimals, key=lambda x: x[0])
    best_precision = max(precision_optimals, key=lambda x: x[0])
    best_recall = max(precision_optimals, key=lambda x: x[0])
    best_hinge_loss = min(hinge_loss_optimals, key=lambda x: x[0])
    best_f1 = max(f1_optimals, key=lambda x: x[0])

    return {
        "accuracy": {"score": best_accuracy[0], "C": best_accuracy[1], "gamma": best_accuracy[2], "kernel": best_accuracy[3]},
        "precision": {"score": best_precision[0], "C": best_precision[1], "gamma": best_precision[2], "kernel": best_precision[3]},
        "recall": {"score": best_recall[0], "C": best_recall[1], "gamma": best_recall[2], "kernel": best_recall[3]},
        "hinge_loss": {"score": best_hinge_loss[0], "C": best_hinge_loss[1], "gamma": best_hinge_loss[2], "kernel": best_hinge_loss[3]},
        "f1": {"score": best_f1[0], "C": best_f1[1], "gamma": best_f1[2], "kernel": best_f1[3]}
    }


def grid_search_rf(clf, n_estimators, max_depths, min_samples_splits, min_samples_leafs, max_features, X_train, y_train, X_test, y_test):
    optimals_list = []
    total_iterations = len(n_estimators)*len(max_depths)*len(min_samples_splits)*len(min_samples_leafs)*len(max_features)
    iteration=1
    fold_amount = 5
    skf = StratifiedKFold(n_splits=fold_amount)
    for n_estimator_value in n_estimators:
        for max_depth_value in max_depths:
            for min_samples_split_value in min_samples_splits:
                for min_samples_leaf_value in min_samples_leafs:
                    for max_feature_value in max_features:
                        print("Iteration {}/{} - n_estimators: {}, max_depth: {}, min_samples_split: {}, min_samples_leaf: {}, max_features: {}".format(iteration, total_iterations, n_estimator_value, max_depth_value, min_samples_split_value, min_samples_leaf_value, max_feature_value))
                        params = {
                            'max_depth': max_depth_value,
                            'n_estimators': n_estimator_value,
                            'min_samples_split': min_samples_split_value,
                            'min_samples_leaf': min_samples_leaf_value,
                            'max_features': max_feature_value
                        }
                        clf.set_params(**params)

                        f = lambda x: x/fold_amount
                        y_scores = np.array([0]*5)
                        y_preds = []
                        pred_pos_list = []

                        for train_index, test_index in skf.split(X_train, y_train):
                            folded_X_train = np.array(X_train)[test_index]
                            folded_y_train = np.array(y_train)[test_index]
                            clf.fit(folded_X_train, folded_y_train)
                            y_pred = clf.predict(X_test)
                            y_scores = y_scores + np.array(get_score(y_test, y_pred))
                            pred_pos = len(list(filter(lambda x: x==1, y_pred)))
                            y_preds.append(y_pred)
                            pred_pos_list.append(pred_pos)

                        print(np.array(pred_pos).mean())
                        if np.array(pred_pos).mean() >= 7:
                            optimals_list.append([f(y_scores), n_estimator_value, max_depth_value, min_samples_split_value, min_samples_leaf_value, max_feature_value])
                        iteration = iteration + 1

    accuracy_optimals = list(map(lambda x: [x[0][0]] + x[1:], optimals_list))
    precision_optimals = list(map(lambda x: [x[0][1]] + x[1:], optimals_list))
    recall_optimals = list(map(lambda x: [x[0][2]] + x[1:], optimals_list))
    hinge_loss_optimals = list(map(lambda x: [x[0][3]] + x[1:], optimals_list))
    f1_optimals = list(map(lambda x: [x[0][4]] + x[1:], optimals_list))

    best_accuracy = max(accuracy_optimals, key=lambda x: x[0])
    best_precision = max(precision_optimals, key=lambda x: x[0])
    best_recall = max(precision_optimals, key=lambda x: x[0])
    best_hinge_loss = max(hinge_loss_optimals, key=lambda x: x[0])
    best_f1 = max(f1_optimals, key=lambda x: x[0])

    return {
        "accuracy": {"score": best_accuracy[0], "max_depth": best_accuracy[1], "n_estimators": best_accuracy[2], "min_samples_splits": best_accuracy[3], "min_samples_leafs": best_accuracy[4], "max_features": best_accuracy[5]},
        "precision": {"score": best_precision[0], "max_depth": best_precision[1], "n_estimators": best_precision[2], "min_samples_splits": best_precision[3], "min_samples_leafs": best_accuracy[4], "max_features": best_accuracy[5]},
        "recall": {"score": best_recall[0], "max_depth": best_recall[1], "n_estimators": best_recall[2], "min_samples_splits": best_recall[3], "min_samples_leafs": best_accuracy[4], "max_features": best_accuracy[5]},
        "hinge_loss": {"score": best_hinge_loss[0], "max_depth": best_hinge_loss[1], "n_estimators": best_hinge_loss[2], "min_samples_splits": best_hinge_loss[3], "min_samples_leafs": best_accuracy[4], "max_features": best_accuracy[5]},
        "f1": {"score": best_f1[0], "max_depth": best_f1[1], "n_estimators": best_f1[2], "min_samples_splits": best_f1[3], "min_samples_leafs": best_accuracy[4], "max_features": best_accuracy[5]}
    }
