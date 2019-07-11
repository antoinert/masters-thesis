import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, hinge_loss, f1_score

from preprocessing import remove_unnecessary_columns, pad_with_zero_rows, remove_after_churn, customer_over_weeks, read_and_filter_table, get_inspected_timeframe, import_and_preprocess_table
from helpers import plot_to_file, pretty_print_scores, log_to_csv, print_feature_importances, plot_svm, pretty_print_scores_svm, training_curve, grid_search_rf

pd.set_option('mode.use_inf_as_na', True)

#F1 optimized:
# {
#     'max_depth': 32/5,
#     'max_features': 34,
#     'min_samples_leaf': 0.1,
#     'min_samples_split': 0.6,
#     'n_estimators': 64/200
# }

#Precision optimized:
# {
#     'max_depth': 20,
#     'max_features': 34,
#     'min_samples_leaf': 0.1,
#     'min_samples_split': 0.1,
#     'n_estimators': 100
# }
# {'max_depth': 20, 'max_features': 34, 'min_samples_leaf': 0.1, 'min_samples_split': 0.2, 'n_estimators': 200}
# {'max_depth': 10, 'max_features': 34, 'min_samples_leaf': 0.1, 'min_samples_split': 0.30000000000000004, 'n_estimators': 64}
n_estimators = [8, 16, 32, 64, 100, 200]
max_depth = range(1, 32)#[5, 10, 20, 32]
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
max_featuress = range(1,34)#[22, 34]
hyperparams = {
    'max_depth': max_depth,
    'n_estimators': n_estimators,
    'min_samples_split': min_samples_splits,
    'min_samples_leaf': min_samples_leafs,
    'max_features': max_featuress
}

def optimize_rf_hyperparameters(filters, train_sizes=[15, 100, 300, 500, 800], table_folder="/", save_file=None, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
    
    X_train = list(map(lambda x: x.flatten(), X_train))
    X_test = list(map(lambda x: x.flatten(), X_test))

    clf = RandomForestClassifier()

    grid_result = grid_search_rf(clf, n_estimators, max_depth, min_samples_splits, min_samples_leafs, max_featuress, X_train, y_train, X_test, y_test)
    print(grid_result)

def rf_run(filters, n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="auto", train_sizes=[15, 100, 300, 500, 800], epochs=15, table_folder="/", save_file=None, input_dim = 34, batch_size = 32, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)

    X_train = list(map(lambda x: x.flatten(), X_train))
    X_test = list(map(lambda x: x.flatten(), X_test))
    
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features)

    # train_sizes, train_scores, validation_scores = learning_curve(clf, X_train, y_train, train_sizes=train_sizes, cv=5, shuffle=True, scoring='f1')
    train_sizes, train_scores, validation_scores = training_curve(clf, X_train, y_train, X_test, y_test, train_sizes=train_sizes, shuffle=True, scoring='precision', train_last=True)
    # exit()
    # print(train_sizes)
    # print(train_scores)
    # print(validation_scores)
    clf.fit(X_train, y_train)

    # cross_val_score(clf, X_train, y_train, scoring='recall_macro', cv=5) 

    y_pred = clf.predict(X_test)
    feature_importances = clf.feature_importances_
    scores = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), hinge_loss(y_test, y_pred), f1_score(y_test, y_pred)]
    # print_feature_importances(feature_importances, feature_names)
    # exit()
    print(len(y_pred))
    print(len(y_test))
    return [y_pred, y_test, feature_importances, scores, train_sizes, train_scores, validation_scores, churn_number, total_number, feature_names]


# {
#     'max_depth': 32/5,
#     'max_features': 34,
#     'min_samples_leaf': 0.1,
#     'min_samples_split': 0.6,
#     'n_estimators': 64/200
# }

#F1-no oversample opt
# {
#     'max_depth': 5,
#     'max_features': 34,
#     'min_samples_leaf': 0.1,
#     'min_samples_split': 0.1,
#     'n_estimators': 8
# }


# 'max_depth': 20,
#     'max_features': 34,
#     'min_samples_leaf': 0.1,
#     'min_samples_split': 0.1,
#     'n_estimators': 100

def process_rfs():
    networks = pd.DataFrame(columns=["y_pred", "y_test", "feature_importances", "scores", "train_sizes", "train_scores", "validation_scores", "churn_number", "total_number", "feature_names"])

    networks.loc["RF_1"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
    networks.loc["RF_2"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
    networks.loc["RF_3"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
    networks.loc["RF_4"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
    networks.loc["RF_5"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
    networks.loc["RF_6"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
    networks.loc["RF_7"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
    networks.loc["RF_8"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
    networks.loc["RF_9"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
    networks.loc["RF_10"] = rf_run(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        train_sizes=[15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800],
                                                        max_depth=14,
                                                        n_estimators=200,
                                                        min_samples_split=0.30000000000000004,
                                                        min_samples_leaf=0.1,
                                                        max_features=28,
                                                        oversample=True)
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

    print(len(train_sizes))
    print(len(train_scores))

    plot_svm(train_sizes, train_scores, validation_scores, save_file="randomforest_optimized_precision_final")
    pretty_print_scores_svm(scores, history_names, churn_numbers, total_numbers, y_preds, y_tests, feature_importances, feature_names)

    # plot_to_file(histories=histories, history_names=history_names, plot_types=["acc", "loss"], algorithm="lstm", save_file="test")
    # pretty_print_scores(scores, history_names, churn_numbers, total_numbers)
    

process_rfs()
# optimize_rf_hyperparameters(["min_amount_of_rows(12)", "no_pure_churn_data"],
#                             table_folder="from_2018-01-15/weekly",
#                             oversample=True)