import dill as pickle 
import pandas as pd
import numpy as np
import os

bbox_df = pd.read_pickle(f'{os.path.dirname(__file__)}/bbox_df.pkl')

param_df = pd.read_pickle(f'{os.path.dirname(__file__)}/slant_param_df.pkl')
unit_vec = param_df.iloc[0]['unit_vec']

with open(f'{os.path.dirname(__file__)}/bounds_slant.pkl', 'br') as f:
    bounds_slant = pickle.load(f)
    
with open(f'{os.path.dirname(__file__)}/in_slant_box.pkl', 'br') as f:
    in_slant_box = pickle.load(f)

def point_in_slant(xyz):
    row = {'point_x': xyz[0],
           'point_y': xyz[1],
           'point_z': xyz[2],
           }
    return in_slant_box(row, unit_vec)