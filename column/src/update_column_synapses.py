import os
import pandas as pd
import numpy as np
import tqdm.auto as tqdm
from annotationframeworkclient import FrameworkClient
import multiwrapper.multiprocessing_utils as mu
import warnings
import click

warnings.filterwarnings("ignore")

num_batches = 100
n_threads = 10

datastack = 'minnie65_phase3_v1'
data_dir = 'data/middle'
synapse_table = 'synapses_pni_2'
soma_table = 'nucleus_detection_v0'
ct_column_table = 'allen_v1_column_types_v2'

qt_dict = {'pre': 'pre_pt_root_id', 'post': 'post_pt_root_id'}


EXCLUDE_CLASSES = ['aibs_coarse_nonneuronal',
                   'aibs_coarse_error',
                   'aibs_coarse_unclear']


def set_valence(x):
    if x == 'aibs_coarse_excitatory':
        return 'Exc'
    elif x in 'aibs_coarse_inhibitory':
        return 'Inh'
    else:
        return 'Uns'


def get_single_soma_df(client, soma_table):
    soma_df = client.materialize.query_table(soma_table)
    soma_df = soma_df.drop(soma_df[soma_df['pt_root_id'] == 0].index)
    soma_count_df = soma_df[['pt_root_id', 'volume']].groupby(
        'pt_root_id').count().reset_index().rename(columns={'volume': 'num_soma'})
    soma_df_clean = soma_df.merge(soma_count_df.query(
        'num_soma == 1'), on='pt_root_id').drop(columns='num_soma')
    return soma_df_clean


def get_unitary_column_df(client,
                          column_table,
                          soma_df,
                          exclude_classes=EXCLUDE_CLASSES):
    column_df_raw = client.materialize.query_table(ct_column_table)
    column_df = column_df_raw.query(
        'classification_system not in @exclude_classes').reset_index(drop=True)

    column_df['valence'] = column_df['classification_system'].apply(
        set_valence)
    column_df = column_df.drop(columns='id').merge(soma_df[[
        'pt_root_id', 'id']], how='inner', on='pt_root_id').rename(columns={'id': 'soma_id'})
    return column_df


def assemble_synapse_df(arg):
    root_ids, client, synapse_table, valence = arg
    return _assemble_synapse_df(root_ids, client, synapse_table, valence)


def _assemble_synapse_df(root_id, client, synapse_table, valence):
    syn_df = client.materialize.query_table(
        synapse_table, filter_equal_dict={qt_dict[valence]: root_id})
    syn_df['syn_depth'] = syn_df['ctr_pt_position'].apply(
        lambda x: x[1] * 4/1000)
    keep_cols = ['pre_pt_root_id', 'post_pt_root_id',
                 'ctr_pt_position', 'size', 'syn_depth']
    return syn_df[keep_cols]


@click.command()
@click.option('-v', '--valence', default='post', help='"pre" or "post", determines which side of the synapses one gets')
def update_column_synapses(valence):
    client = FrameworkClient(datastack)
    version = client.materialize.version
    filename = f'{data_dir}/column_synapses_v{version}_{valence}.feather'

    if os.path.exists(filename):
        print(f'\tData already exists at {filename}!')
    else:
        print('\tAcquiring soma information...')
        soma_df = get_single_soma_df(client, soma_table)
        column_df = get_unitary_column_df(client, ct_column_table, soma_df)

        col_root_ids = column_df['pt_root_id'].values
        args = [(int(root_id), client, synapse_table, valence)
                for root_id in col_root_ids]
        print(f'\tStarting synapse download for {len(args)} cells...')

        syn_df_chunked = []
        for arg_batch in tqdm.tqdm(np.array_split(args, num_batches)):
            sdf = mu.multiprocess_func(
                assemble_synapse_df, list(arg_batch), n_threads=n_threads)
            syn_df_chunked.append(pd.concat(sdf).reset_index(drop=True))

        syn_df = pd.concat(syn_df_chunked).reset_index(drop=True)
        syn_df.to_feather(filename)
        print(f'\tFinished saving to {filename}')


if __name__ == '__main__':
    update_column_synapses()
