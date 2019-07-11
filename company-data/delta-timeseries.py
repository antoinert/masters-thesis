import pandas as pd
import numpy as np

master_table = pd.read_csv('../master_table.csv', nrows=10896)
master_table.drop('has_churned_y', 1, inplace=True)
master_table.drop('has_churned_y.1', 1, inplace=True)
master_table.drop('has_churned_y.2', 1, inplace=True)
master_table.drop('has_churned_y.3', 1, inplace=True)
master_table.drop('has_churned_x.1', 1, inplace=True)
master_table.drop('has_churned_x.2', 1, inplace=True)
master_table.drop('has_churned_x.3', 1, inplace=True)
master_table.drop('total_spent_eurocents_y', 1, inplace=True)
master_table.drop('total_spent_eurocents', 1, inplace=True)

master_table = master_table.fillna(0)
print(master_table)
master_table = master_table.drop(columns='date_1').iloc[0:26]
print(master_table)
master_table = [ frame.drop(columns='smartly_company_id').iloc[:, 2:].pct_change().shift(0) for company, frame in master_table.groupby('smartly_company_id')]
# print(master_table.groupby('smartly_company_id'))#.pct_change().shift(0))
print(master_table)
# print(master_table.iloc[:, 4:].pct_change().shift(0))