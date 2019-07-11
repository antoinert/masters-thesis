import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, validation_curve, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, hinge_loss, f1_score

from preprocessing import remove_unnecessary_columns, pad_with_zero_rows, remove_after_churn, customer_over_weeks, read_and_filter_table, get_inspected_timeframe, import_and_preprocess_table
from helpers import plot_to_file, pretty_print_scores, log_to_csv, print_feature_importances, plot_svm, pretty_print_scores_svm, training_curve, grid_search_svm

pd.set_option('mode.use_inf_as_na', True)

gammas = np.concatenate((np.linspace(0.01, 0.1, 10, endpoint=True), np.linspace(0.2, 1.0, 9, endpoint=True)), axis=0)
c_ranges = np.concatenate((np.linspace(0.01, 0.1, 10, endpoint=True), np.linspace(0.2, 1.0, 9, endpoint=True)), axis=0)
kernels = ['rbf']
hyperparams = {
    'C': c_ranges,
    'gamma': gammas,
    'kernel': kernels
}

def optimize_svm_hyperparameters(filters, train_sizes=[15, 100, 300, 500, 800], table_folder="/", save_file=None, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
    
    X_train = list(map(lambda x: x.flatten(), X_train))
    X_test = list(map(lambda x: x.flatten(), X_test))

    clf = svm.SVC()

    grid_result = grid_search_svm(clf, c_ranges, gammas, kernels, X_train, y_train, X_test, y_test)
    print(grid_result)
    # clf.fit(X_train, y_train)

    # print(clf.best_params_)

def svm_run(filters, c_range=1.0, kernel_type='rbf', gamma='auto', train_sizes=[15, 100, 300, 500, 800], table_folder="/", save_file=None, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
    
    X_train = list(map(lambda x: x.flatten(), X_train))
    X_test = list(map(lambda x: x.flatten(), X_test))

    clf = svm.SVC(kernel=kernel_type, gamma=gamma, C=c_range)

    # train_sizes, train_scores, validation_scores = learning_curve(clf, X_train, y_train, train_sizes=train_sizes, cv=5, shuffle=True, scoring='f1')
    train_sizes, train_scores, validation_scores = training_curve(clf, X_train, y_train, X_test, y_test, train_sizes=train_sizes, shuffle=True, scoring='precision', train_last=True)

    # print(train_scores, valid_scores)
    clf.fit(X_train, y_train)

    # cross_val_score(clf, X_train, y_train, scoring='recall_macro', cv=5) 
    y_pred = clf.predict(X_test)
    if kernel_type == 'linear':
        feature_importances = clf.coef_.flatten()
    else:
        feature_importances = []
    scores = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), hinge_loss(y_test, y_pred), f1_score(y_test, y_pred)]
    
    # print_feature_importances(feature_importances, feature_names, all=True)
    print(y_pred)
    return [y_pred, y_test, feature_importances, scores, train_sizes, train_scores, validation_scores, churn_number, total_number, feature_names]


# Ni = 34
# No = 1
# a = 2
# Ns = 1212

#1212/(5*(35)) = 7 hidden layers

def process_svms():
    networks = pd.DataFrame(columns=["y_pred", "y_test", "feature_importances", "scores", "train_sizes", "train_scores", "validation_scores", "churn_number", "total_number", "feature_names"])

    networks.loc["SVM_1"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    networks.loc["SVM_2"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    networks.loc["SVM_3"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    networks.loc["SVM_4"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    networks.loc["SVM_5"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    networks.loc["SVM_6"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    networks.loc["SVM_7"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    networks.loc["SVM_8"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    networks.loc["SVM_9"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    networks.loc["SVM_10"] = svm_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        c_range=0.9,
                                                        gamma=0.2)
    
    y_preds = networks.y_pred
    y_tests = networks.y_test
    feature_importances = networks.feature_importances
    scores = networks.scores
    train_sizes = networks.train_sizes
    train_scores = networks.train_scores
    validation_scores = networks.validation_scores
    churn_numbers = networks.churn_number
    total_numbers = networks.total_number
    history_names = networks.index
    feature_names = networks.feature_names

    plot_svm(train_sizes, train_scores, validation_scores, save_file="optimized_f1_final")
    pretty_print_scores_svm(scores, history_names, churn_numbers, total_numbers, y_preds, y_tests, feature_importances, feature_names)
    

process_svms()
# optimize_svm_hyperparameters(["min_amount_of_rows(12)", "no_pure_churn_data"],
#                             table_folder="from_2018-01-15/weekly",
#                             oversample=True)