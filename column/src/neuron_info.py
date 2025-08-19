import numpy as np
import pandas as pd

voxel_resolution = [4, 4, 40]


def get_syn_df(oid, synapse_table, client, pre=True, post=True, remove_self_synapse=True):
    if pre:
        pre_syn_df = client.materialize.query_table(synapse_table, filter_equal_dict={
                                                    'pre_pt_root_id': np.int(oid)})
    else:
        pre_syn_df = pd.DataFrame()
    if post:
        post_syn_df = client.materialize.query_table(synapse_table, filter_equal_dict={
                                                     'post_pt_root_id': np.int(oid)})
    else:
        post_syn_df = pd.DataFrame()

    syn_df = pd.concat((pre_syn_df, post_syn_df))
    if remove_self_synapse:
        syn_df = syn_df.query(
            'pre_pt_root_id != post_pt_root_id').reset_index(drop=True)
    return syn_df


def syn_depth(syn_df, voxel_resolution):
    return syn_df['ctr_pt_position'].apply(lambda x: x[1]*voxel_resolution[1]/1000)


def neuron_info(oid, cell_type_df, voxel_resolution=voxel_resolution):
    try:
        nrn_dat = cell_type_df.query(
            'pt_root_id == @oid', engine='python')
        if len(nrn_dat) != 1:
            return None, None, None
        if 'pt_position' in nrn_dat.columns:
            soma_position = nrn_dat['pt_position'].values[0] * voxel_resolution
        else:
            soma_position = None

        if 'soma_id' in nrn_dat.columns:
            nrn_soma_id = nrn_dat['soma_id'].values[0]
        else:
            nrn_soma_id = None

        if 'cell_type' in nrn_dat.columns:
            nrn_subtype = nrn_dat['cell_type'].values[0]
        else:
            nrn_subtype = None
    except:
        soma_position = None
        nrn_soma_id = None
        nrn_subtype = None
    return nrn_subtype, nrn_soma_id, soma_position
