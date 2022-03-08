""" read results """
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import argparse
import os
import pickle

import pandas as pd

from im2mesh import config

parser = argparse.ArgumentParser(
    description='process results'
)
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

testset = cfg['data']['classes'][0]
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_file = os.path.join(generation_dir, f'eval_meshes_full_{testset}.pkl')

# load pickle
with open(out_file, 'rb') as reader:
    results = pickle.load(reader)

# don't shorten any column or row with ...
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

select_columns = {
    'fscore@1 (mesh)': 1, # 1 = number of decimal digits
    'chamfer (mesh)': 1,
    'iou (mesh)': 1,
    # TODO enable invis
    'fscore@1_vis (mesh)': 1,
    'chamfer_vis (mesh)': 1,
    'fscore@1_invis (mesh)': 1,
    'chamfer_invis (mesh)': 1,
}

print(list(select_columns.keys()))
print(type(select_columns.keys()))

## show sorted
results = results.sort_values('fscore@1 (mesh)')
print(results[['modelname'] + list(select_columns.keys())])
## show small range
# small_scores = results.loc[(results['fscore@1 (mesh)'] >= 0.05) & (results['fscore@1 (mesh)'] <= 0.1)]
# print(small_scores[['modelname', 'fscore@1 (mesh)', 'fscore@1_vis (mesh)', 'fscore@1_invis (mesh)', 'iou (mesh)']])
## show median
# print(results.loc[:,'fscore@1 (mesh)'].median())
# print("Median:")
# print(results.median(numeric_only=True))#, level='class name'))
print("Class Max")
# print(results.max(numeric_only=True))#, level='class name'))
print(results[['class name'] + list(select_columns.keys())].groupby('class name').max(numeric_only=True))
print()
print("Class Min")
# print(results.min(numeric_only=True))#, level='class name'))
print(results[['class name'] + list(select_columns.keys())].groupby('class name').min(numeric_only=True))
print()

# show class mean
print("Class Mean")
eval_df_class = results.groupby(by=['class name']).mean()
eval_df_class.loc['mean'] = eval_df_class.mean()
eval_df_class.loc['mean_sample'] = results.mean()
print(eval_df_class[list(select_columns.keys())])

print()
print("--------------------------------------------------")
print("Latex table format")
print("Chamfer * 100, Fscore * 100")
# latex_table = (eval_df_class[['fscore@1 (mesh)', 'chamfer (mesh)', 'iou (mesh)']] * [100, 100, 100]).round({'fscore@1 (mesh)':1, 'chamfer (mesh)':1, 'iou (mesh)':1})
latex_table = (eval_df_class[list(select_columns.keys())] * [100 for i in range(len(select_columns))]).round(select_columns)
latex_table['combined'] = ''
latex_table['combined'] = latex_table['fscore@1 (mesh)'].astype(str) + " & " + latex_table['chamfer (mesh)'].astype(str)
print(latex_table.T)
print("--------------------------------------------------")

# real latex table
print()
print(latex_table.T.to_latex())
# print(latex_table.T.to_latex(float_format='{:.2f}'.format))

# to csv for easier mean unseen computation
print()
print(latex_table.T.to_csv())
