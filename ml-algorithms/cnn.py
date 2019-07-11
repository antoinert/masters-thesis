import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten
from keras.callbacks import BaseLogger, EarlyStopping
import keras_metrics

from sklearn.model_selection import train_test_split

from preprocessing import remove_unnecessary_columns, pad_with_zero_rows, remove_after_churn, customer_over_weeks, customer_under_weeks, read_and_filter_table, get_inspected_timeframe, import_and_preprocess_table
from helpers import plot_to_file, pretty_print_scores, log_to_csv

pd.set_option('mode.use_inf_as_na', True)

def cnn(filters, pooling_size=2, epochs=15, table_folder="/", kernel_size = 3, input_dim = 34, batch_size = 32, nb_filters = 34, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to
    
    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)

    print("Creating layers...")
    model = Sequential()
    model.add(Conv1D(nb_filters, kernel_size=kernel_size, input_shape=(timesteps, input_dim), activation='relu'))
    model.add(MaxPooling1D(pooling_size))
    # model.add(Conv1D(nb_filters, kernel_size=kernel_size, activation='relu'))
    # model.add(Conv1D(nb_filters*2, kernel_size=3, activation='relu'))
    # model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='loss', patience=5)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict(X_test)

    log_to_csv("cnn", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json(), nb_filters, kernel_size)

    return [score, history, churn_number, total_number, y_pred]

def cnn2(filters, pooling_size=2, epochs=15, table_folder="/", kernel_size = 3, input_dim = 34, batch_size = 32, nb_filters = 34, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to
    
    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)

    print("Creating layers...")
    model = Sequential()
    model.add(Conv1D(nb_filters, kernel_size=kernel_size, input_shape=(timesteps, input_dim), activation='relu'))
    model.add(MaxPooling1D(pooling_size))
    # model.add(Conv1D(nb_filters, kernel_size=kernel_size, activation='relu'))
    # model.add(Conv1D(nb_filters*2, kernel_size=3, activation='relu'))
    # model.add(Conv1D(nb_filters*2, kernel_size=3, activation='relu'))
    # model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='loss', patience=2)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict(X_test)

    log_to_csv("cnn", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json(), nb_filters, kernel_size)

    return [score, history, churn_number, total_number, y_pred]

def cnn3(filters, pooling_size=2, epochs=15, table_folder="/", kernel_size = 3, input_dim = 34, batch_size = 32, nb_filters = 34, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to
    
    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)

    print("Creating layers...")
    model = Sequential()
    model.add(Conv1D(nb_filters, kernel_size=kernel_size, input_shape=(timesteps, input_dim), activation='relu'))
    model.add(MaxPooling1D(pooling_size))
    model.add(Conv1D(nb_filters, kernel_size=kernel_size, activation='relu'))
    model.add(Conv1D(nb_filters, kernel_size=kernel_size, activation='relu'))
    # model.add(Conv1D(nb_filters*2, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='loss', patience=2)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)

    y_pred = model.predict(X_test)

    log_to_csv("cnn", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json(), nb_filters, kernel_size)

    return [score, history, churn_number, total_number, y_pred]

def process_cnns():
    networks = pd.DataFrame(columns=["score", "history", "churn_number", "total_number", "y_pred"])
   
    networks.loc["CNN_1"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
                                                            table_folder="from_2018-01-15/weekly",
                                                            epochs=100,
                                                            oversample=True)  
    # networks.loc["CNN_2"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_3"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)  
    # networks.loc["CNN_4"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)  
    # networks.loc["CNN_5"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)  
    # networks.loc["CNN_6"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)  
    # networks.loc["CNN_7"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)  
    # networks.loc["CNN_8"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)  
    # networks.loc["CNN_9"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)  
    # networks.loc["CNN_10"] = cnn(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)  

    # networks.loc["CNN_2layers"] = cnn2(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_3layers"] = cnn3(["no_pure_churn_data", "min_amount_of_rows(12)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_2"] = cnn(["no_pure_churn_data", "min_amount_of_rows(8)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_3"] = cnn(["no_pure_churn_data", "min_amount_of_rows(8)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_4"] = cnn(["no_pure_churn_data", "min_amount_of_rows(8)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_5"] = cnn(["no_pure_churn_data", "min_amount_of_rows(8)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_6"] = cnn(["no_pure_churn_data", "min_amount_of_rows(8)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_7"] = cnn(["no_pure_churn_data", "min_amount_of_rows(8)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_8"] = cnn(["no_pure_churn_data", "min_amount_of_rows(8)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_9"] = cnn(["no_pure_churn_data", "min_amount_of_rows(8)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    # networks.loc["CNN_10"] = cnn(["no_pure_churn_data", "min_amount_of_rows(8)"],
    #                                                         table_folder="from_2018-01-15/weekly",
    #                                                         epochs=100,
    #                                                         oversample=True)
    
    histories = networks.history
    scores = networks.score
    history_names = networks.index
    churn_numbers = networks.churn_number
    total_numbers = networks.total_number
    y_preds = networks.y_pred


    plot_to_file(histories=histories, history_names=history_names, plot_types=["acc", "loss", "precision", "recall", "f1_score"], algorithm="cnn", save_file="50epochs_optimized")
    pretty_print_scores(scores, history_names, churn_numbers, total_numbers, y_preds)


process_cnns()
    