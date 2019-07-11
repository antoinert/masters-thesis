import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import BaseLogger, EarlyStopping
import keras_metrics

from sklearn.model_selection import StratifiedKFold

from preprocessing import remove_unnecessary_columns, pad_with_zero_rows, remove_after_churn, customer_over_weeks, read_and_filter_table, get_inspected_timeframe, import_and_preprocess_table
from helpers import plot_to_file, pretty_print_scores, log_to_csv

pd.set_option('mode.use_inf_as_na', True)


def lstm(filters, epochs=15, table_folder="/", save_file=None, input_dim = 34, batch_size = 32, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
    
    print("Creating layers...")

    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    scores = []
    histories = []
    churn_numbers = []
    total_numbers = []
    history_names = [i for i in range(0, kfold.get_n_splits())]

    for train, test in kfold.split(np.zeros(len(y_train)), y_train):
        model = Sequential()
        model.add(LSTM(input_dim, input_length=timesteps, input_dim=input_dim, return_sequences=True))
        model.add(LSTM(input_dim))
        model.add(Dense(1, activation='sigmoid'))
        print("Compiling model...")
        model.compile(loss='mean_squared_error',
                    optimizer='rmsprop',
                    metrics=['accuracy'])
        print("Fitting model...")
        print(model.summary())
        callback = [
            EarlyStopping(monitor='val_loss', patience=5)
        ]
        
        history = model.fit(X_train[test], y_train[test], validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
        score = model.evaluate(X_test, y_test, batch_size=batch_size)
        scores.append(score)
        histories.append(history)
        churn_numbers.append(churn_number)
        total_numbers.append(total_number)
        log_to_csv("lstm", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json())

    plot_to_file(histories=histories, history_names=history_names, plot_types=["acc", "loss"], algorithm="lstm", save_file="regular")
    pretty_print_scores(scores, history_names, churn_numbers, total_numbers)

    # return [score, history, churn_number, total_number]


def lstm2(filters, epochs=15, table_folder="/", save_file=None, input_dim = 34, batch_size = 32, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
        
    print("Creating layers...")

    model = Sequential()
    model.add(LSTM(34, input_length=timesteps, input_dim=34, return_sequences=True))
    model.add(LSTM(input_dim))
    model.add(Dense(1, activation='sigmoid'))
    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='val_loss', patience=5)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict(X_test)

    log_to_csv("lstm", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json())

    return [score, history, churn_number, total_number, y_pred]


def lstm3(filters, epochs=15, table_folder="/", save_file=None, input_dim = 34, batch_size = 32, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
        
    print("Creating layers...")

    model = Sequential()
    model.add(LSTM(34, input_length=timesteps, input_dim=34, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='val_loss', patience=5)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict(X_test)

    log_to_csv("lstm", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json())

    return [score, history, churn_number, total_number, y_pred]


def lstm4(filters, epochs=15, table_folder="/", save_file=None, input_dim = 34, batch_size = 32, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
        
    print("Creating layers...")

    model = Sequential()
    model.add(LSTM(34, input_length=timesteps, input_dim=34, return_sequences=True))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(LSTM(input_dim))
    model.add(Dense(1, activation='sigmoid'))
    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='val_loss', patience=5)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict(X_test)

    log_to_csv("lstm", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json())

    return [score, history, churn_number, total_number, y_pred]

def lstm5(filters, epochs=15, table_folder="/", save_file=None, input_dim = 34, batch_size = 32, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
        
    print("Creating layers...")

    model = Sequential()
    model.add(LSTM(34, input_length=timesteps, input_dim=34, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    # model.add(LSTM(input_dim, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(input_dim, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='val_loss', patience=5)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict(X_test)

    log_to_csv("lstm", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json())

    return [score, history, churn_number, total_number, y_pred]

def lstm6(filters, epochs=15, table_folder="/", save_file=None, input_dim = 34, batch_size = 32, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
        
    print("Creating layers...")

    model = Sequential()
    model.add(LSTM(34, input_length=timesteps, input_dim=input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='val_loss', patience=5)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict(X_test)

    log_to_csv("lstm", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json())

    return [score, history, churn_number, total_number, y_pred]

def lstm7(filters, epochs=15, table_folder="/", save_file=None, input_dim = 34, batch_size = 32, time_from = 32, time_to = 8, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)
        
    print("Creating layers...")

    model = Sequential()
    model.add(LSTM(34, input_length=timesteps, input_dim=input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='val_loss', patience=5)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    y_pred = model.predict(X_test)

    log_to_csv("lstm", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json())

    return [score, history, churn_number, total_number, y_pred]
# Ni = 34
# No = 1
# a = 2
# Ns = 1212

#1212/(5*(35)) = 7 hidden layers

def process_lstms():
    networks = pd.DataFrame(columns=["score", "history", "churn_number", "total_number", "y_pred"])

    networks.loc["LSTM_1"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)
    networks.loc["LSTM_2"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)
    networks.loc["LSTM_3"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)
    networks.loc["LSTM_4"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)
    networks.loc["LSTM_5"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)
    networks.loc["LSTM_6"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)
    networks.loc["LSTM_7"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)
    networks.loc["LSTM_8"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)
    networks.loc["LSTM_9"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)
    networks.loc["LSTM_10"] = lstm2(["min_amount_of_rows(12)", "no_pure_churn_data"],
                                                        table_folder="from_2018-01-15/weekly",
                                                        oversample=True,
                                                        epochs=100)


    histories = networks.history
    scores = networks.score
    history_names = networks.index
    churn_numbers = networks.churn_number
    total_numbers = networks.total_number
    y_preds = networks.y_pred

    plot_to_file(histories=histories, history_names=history_names, plot_types=["acc", "loss", "precision", "recall", "f1_score"], algorithm="lstm", save_file="50epochs_optimized")
    pretty_print_scores(scores, history_names, churn_numbers, total_numbers, y_preds)
    

process_lstms()