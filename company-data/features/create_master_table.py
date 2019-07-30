
import sys
import glob
import pandas as pd

pd.set_option('mode.chained_assignment', None)

def create_master_table(folder):
    filenames = glob.glob('./{}/*.csv'.format(folder))
    df_list_feature = []

    for filename in filenames:
        if 'master_table.csv' not in filename:
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

    master_table = preprocess_df(master_table)

    print("Number of lines: {}".format(len(master_table)))
    master_table.to_csv('./{}/master_table.csv'.format(folder))


def preprocess_df(master_table):
    master_table = remove_unnecessary_columns(master_table)
    master_table = normalize_occasional_columns(master_table)
    master_table = master_table.groupby('smartly_company_id').filter(lambda df: no_pure_churn_data(df))
    master_table = master_table.fillna(0)
    return master_table

def no_pure_churn_data(df):
    if all(i == 1 for i in df.has_churned_x.unique()):
        return False
    return True

def remove_unnecessary_columns(master_table):
    # drop duplicate columns
    master_table = master_table.loc[:,~master_table.columns.duplicated()]

    # drop any custom column
    master_table.drop('has_churned_y', 1, inplace=True)

    return master_table

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

def normalize_occasional_columns(master_table):
    avg = master_table.avg.fillna(0)
    nps_score = master_table.nps_score.fillna(0)

    avg_index = avg.index
    avg_values = normalize_occasional(avg.values, avg_index)

    nps_score_index = nps_score.index
    nps_score_values = normalize_occasional(nps_score.values, nps_score_index)

    master_table['nps_score'] = nps_score_values
    master_table['avg'] = avg_values

    return master_table

create_master_table(sys.argv[1])