
import sys
import glob
import pandas as pd

def create_master_table(folder):
    filenames = glob.glob('./{}/*.csv'.format(folder))
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

    print("Number of lines: {}".format(len(master_table)))
    master_table.to_csv('./{}/master_table.csv'.format(folder))

create_master_table(sys.argv[1])