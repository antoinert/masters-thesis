import pandas as pd
import numpy as np
import glob

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

def remove_unnecessary_columns(master_table):
    master_table.drop('has_churned_y', 1, inplace=True)
    master_table.drop('has_churned_y.1', 1, inplace=True)
    master_table.drop('has_churned_y.2', 1, inplace=True)
    master_table.drop('has_churned_y.3', 1, inplace=True)
    master_table.drop('has_churned_x.1', 1, inplace=True)
    master_table.drop('has_churned_x.2', 1, inplace=True)
    master_table.drop('has_churned_x.3', 1, inplace=True)
    master_table.drop('total_spent_eurocents_y', 1, inplace=True)
    master_table.drop('total_spent_eurocents', 1, inplace=True)
    master_table.drop('date_1', 1, inplace=True)
    return master_table

def pad_with_zero_rows(dataframe, fixed_length):
    pad = {}
    pad_length = fixed_length - len(dataframe)
    # if len(dataframe) > 33:
    #     print(dataframe)
    for key in dataframe.keys():
        pad[key] = [0] * pad_length

    return pd.concat([pd.DataFrame.from_dict(pad), dataframe], ignore_index=True)

def remove_after_churn(dataframe):
    df_len = len(dataframe)

    for index, row in enumerate(dataframe.values):
        has_churned = row[2]
        if has_churned == 1:
            return dataframe.iloc[0:index]

    return dataframe

def get_inspected_timeframe(dataframe, time_from, time_to):
    df_len = len(dataframe)

    index_start = df_len-time_from if df_len >= time_from else 0
    index_end = -time_to if df_len > time_to else df_len
    shortened_df = dataframe.iloc[index_start:index_end]
    
    return shortened_df

def customer_under_weeks(df, weeks):
    if len(df.has_churned_x[df.has_churned_x == 0]) <= weeks:
        return True
    return False

def customer_over_weeks(df, weeks):
    if len(df.has_churned_x[df.has_churned_x == 0]) >= weeks:
        return True
    return False

def no_pure_churn_data(df):
    if all(i == 1 for i in df.has_churned_x.unique()):
        return False
    return True

def min_amount_of_rows(df, rows):
    return False if len(df) <= rows else True

def exec_filter(filter, df):
    if filter.find("(") != -1:
        split_filter = filter.split("(")
        func = split_filter[0]
        arg = int(split_filter[1].split(")")[0])
        return eval(func)(df, arg)
    else:
        return eval(filter)(df)

def apply_filters(filters, df):
    if not filters:
        return True

    results = list(map(lambda filter: exec_filter(filter, df), filters))
    return all(result == True for result in results)

def read_and_filter_table(filters, table_folder):
    try: 
        table = pd.read_csv('../company-data/features/{}/master_table.csv'.format(table_folder)).sort_values(by=["date_1"])
    except FileNotFoundError:
        create_master_table(table_folder)
        table = pd.read_csv('../company-data/features/{}/master_table.csv'.format(table_folder)).sort_values(by=["date_1"])

    return table.groupby('smartly_company_id').filter(lambda x: apply_filters(filters, x))

def normalize_occasional(values, indexes):
    previous_avg = 0

    for i in range(0, len(values)):
        current_avg = values[i]
        
        if current_avg == 0 and previous_avg != 0:
            current_avg = previous_avg
        else:
            previous_avg = current_avg

        values[i] = current_avg
    
    return pd.Series(values, index=indexes)

def feature_engineer_table(master_table):
    avg = master_table.avg.fillna(0)
    nps_score = master_table.nps_score.fillna(0)
    
    avg_index = avg.index
    avg_values = normalize_occasional(avg.values, avg_index)
    
    nps_score_index = nps_score.index
    nps_score_values = normalize_occasional(nps_score.values, nps_score_index)

    previous_avg = 0

    support_rating = nps_score_values + avg_values

    master_table.drop('avg', 1, inplace=True)
    master_table.drop('nps_score', 1, inplace=True)

    master_table["support_rating"] = support_rating
    return master_table

def downsample(df, ratio):
    grouped = df.groupby('smartly_company_id')
    company_dict = {}
    companies = []
    churns = []
    for company, frame in grouped:
        companies.append(company)
        churns.append(frame.has_churned_x.max())

    company_dict["company"] = companies
    company_dict["churned"] = churns
    
    df_companies = pd.DataFrame(company_dict)
    churned_companies = df_companies[df_companies.churned == 1].company.tolist()
    not_churned_companies = df_companies[df_companies.churned == 0].company.tolist()

    X = int(len(churned_companies)/ratio)

    downsampled_list = resample(not_churned_companies, replace=False, n_samples=X) + churned_companies
    df_downsampled = df[df.smartly_company_id.isin(downsampled_list)]

    return df_downsampled

def oversample(grouped_df, churns):
    company_dict = {}
    companies = []
    frames = {}
    for company, frame in grouped_df:
        companies.append(company)
        frames[company] = frame

    company_dict["company"] = companies
    company_dict["churned"] = churns
    
    df_companies = pd.DataFrame(company_dict)
    churned_companies = df_companies[df_companies.churned == 1]

    
    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(df_companies, df_companies.churned)
    print("Oversampled churned from {} to {}".format(len(churned_companies), len(list(filter(lambda x: x[1] == 1, X_resampled)))))

    oversampled_list = list(map(lambda company: [company[0], frames[company[0]]], X_resampled))
    
    return oversampled_list, y_resampled

def process_dfs(grouped_dfs, time_from, time_to, timesteps):
    X = []
    for company, frame in grouped_dfs:
        before_churn_frame = remove_after_churn(frame)
        
        shortened_frame = get_inspected_timeframe(before_churn_frame, time_from, time_to)
        
        padded_frame = pad_with_zero_rows(shortened_frame, timesteps)

        final_frame = padded_frame.iloc[:, 3:].pct_change().shift(0).fillna(0).values
        X.append(final_frame)

    write_df_to_file(X)
    return X

def write_df_to_file(li):
    print(li)
    df = pd.DataFrame(li)
    df.to_csv('csv_test.csv')

def import_and_preprocess_table(timesteps, time_from, time_to, filters, table_folder, downsample_ratio=None, oversample_data=None):    
    master_table = read_and_filter_table(filters, table_folder)
    master_table = remove_unnecessary_columns(master_table)
    master_table = feature_engineer_table(master_table)
    if downsample_ratio:
        master_table = downsample(master_table, downsample_ratio)
        grouped = master_table.groupby('smartly_company_id')
    else:
        grouped = master_table.groupby('smartly_company_id')
    
    X = []
    y = []

    for company, frame in grouped:
        X.append([company, frame])

    for company, frame in grouped:
        churn_class = frame.has_churned_x.max()
        y.append(churn_class)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    
    original_X_train = np.array(process_dfs(X_train, time_from, time_to, timesteps))
    original_y_train = np.array(y_train)

    if oversample_data:
        X_train, y_train = oversample(X_train, y_train)

    X_processed_train = process_dfs(X_train, time_from, time_to, timesteps)
    X_processed_test = process_dfs(X_test, time_from, time_to, timesteps)

    X_batch_train = np.array(X_processed_train)
    X_batch_test = np.array(X_processed_test)
    y_batch_train = np.array(y_train)
    y_batch_test = np.array(y_test)

    churn_number = len(list(filter(lambda x: x == 1, y_train)))
    total_number = len(y_train)
    non_churn_number = total_number - churn_number

    print("Churn: ", churn_number/total_number)
    print("Non-churn: ", non_churn_number/total_number)
    print("Total: ", total_number)

    return X_batch_train, X_batch_test, y_batch_train, y_batch_test, churn_number, total_number, list(master_table.iloc[:, 3:])

def create_master_table(folder):
    filenames = glob.glob('../company-data/features/{}/*.csv'.format(folder))
    df_list_feature = []

    for filename in filenames:
        df = pd.read_csv(filename)
        print(filename)
        df.set_index('smartly_company_id')
        df_list_feature.append(df)

    def print_sizes(arr):
        for i in arr:
            print(len(i))

    df_list_feature = sorted(df_list_feature, key=len, reverse=True)
    
    master_table = df_list_feature[0]
    i = 1

    while i < len(df_list_feature):
        master_table = master_table.merge(df_list_feature[i], how='left', on=['smartly_company_id', 'date_1'])
        i = i + 1

    print(len(master_table))
    master_table.to_csv('../company-data/features/{}/master_table.csv'.format(folder))