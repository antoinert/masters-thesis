import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, GlobalAveragePooling1D, Dropout, Flatten, LSTM
from keras.callbacks import BaseLogger, EarlyStopping

from sklearn.model_selection import train_test_split

from preprocessing import remove_unnecessary_columns, pad_with_zero_rows, remove_after_churn, customer_over_weeks, read_and_filter_table, get_inspected_timeframe, import_and_preprocess_table
from helpers import plot_to_file, pretty_print_scores, log_to_csv

pd.set_option('mode.use_inf_as_na', True)

def convlstm(filters, epochs=15, table_folder="/", kernel_size = 3, input_dim = 34, batch_size = 32, nb_filters = 34, time_from = 28, time_to = 4, downsample_ratio=None, oversample=None):
    timesteps = time_from-time_to

    X_train, X_test, y_train, y_test, churn_number, total_number, feature_names = import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio, oversample)

    print("Creating layers...")
    model = Sequential()
    model.add(Conv1D(nb_filters, kernel_size=kernel_size, input_shape=(timesteps, input_dim), activation='relu'))
    # model.add(MaxPooling1D(2))
    model.add(LSTM(input_dim, return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(35, return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(20, return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(20, return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(Conv1D(nb_filters, kernel_size=3, activation='relu'))
    model.add(LSTM(input_dim))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    print("Compiling model...")
    model.compile(loss='mean_squared_error',
                optimizer='rmsprop',
                metrics=['accuracy'])
    print("Fitting model...")
    print(model.summary())
    callback = [
        EarlyStopping(monitor='loss', patience=2)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)#, callbacks=callback)
    score = model.evaluate(X_test, y_test, batch_size=batch_size)

    log_to_csv("convlstm", score, history, filters, table_folder, input_dim, batch_size, time_from, time_to, model.to_json(), nb_filters, kernel_size)

    return [score, history, churn_number, total_number]

def process_convlstms():
    networks = pd.DataFrame(columns=["score", "history", "churn_number", "total_number"])
    
    networks.loc["CONVLSTM_1"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)
    networks.loc["CONVLSTM_2"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)
    networks.loc["CONVLSTM_3"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)
    networks.loc["CONVLSTM_4"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)
    networks.loc["CONVLSTM_5"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)
    networks.loc["CONVLSTM_6"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)
    networks.loc["CONVLSTM_7"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)
    networks.loc["CONVLSTM_8"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)
    networks.loc["CONVLSTM_9"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)
    networks.loc["CONVLSTM_10"] = convlstm(["no_pure_churn_data", "min_amount_of_rows(8)"],
                                                                    table_folder="from_2018-01-15/weekly",
                                                                    oversample=True,
                                                                    epochs=100)

    histories = networks.history
    scores = networks.score
    history_names = networks.index
    churn_numbers = networks.churn_number
    total_numbers = networks.total_number

    plot_to_file(histories=histories, history_names=history_names, plot_types=["acc", "loss"], algorithm="convlstm", save_file="100epochs_10sessions")
    pretty_print_scores(scores, history_names, churn_numbers, total_numbers)


process_convlstms()