import pandas as pd, numpy as np
import os
RAW = os.environ.get('P2P_RAW', os.path.join(os.path.dirname(__file__), '..', 'data_raw') + '/')
def load_phase1():
    df = pd.read_csv(RAW+'input_file_1.csv', index_col=0,
        dtype={'dti':'float32','emp_length':'float32','loan_amnt':'float32','purpose':'object','rejected':'int8'})
    df['issue_d']=pd.to_datetime(df['issue_d'])
    return df
def load_phase2():
    df = pd.read_csv(RAW+'input_file_2.csv', index_col=0)
    df['issue_d']=pd.to_datetime(df['issue_d'], format='%b-%Y')
    return df
