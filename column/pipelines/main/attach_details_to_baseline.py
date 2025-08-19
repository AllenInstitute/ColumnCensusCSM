import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import pickle
import dotenv

import os
import sys
import click

from datetime import datetime

sys.path.append(os.path.abspath("../cards"))
from src.synapse_budget import *
import card_config

from common_setup import project_info, project_paths
from affine_transform import minnie_column_transform

tform = minnie_column_transform()

column_file_dir = f"{project_paths.data}/temp"
synapse_file_dir = f"{project_paths.data}/synapse_files"
voxel_resolution = np.array([4, 4, 40])

proximal_threshold = 50 + 15
proximal_threshold = 50 + 7.5

layerConfig = card_config.LayerConfig()


def baseline_data_filename(process_dir, configuration_name, file_tag):
    return f"{process_dir}/{configuration_name}_{file_tag}_files.pkl"


def define_comp(row):
    if row["valence"] == "Inh":
        if row["is_soma"]:
            return "soma"
        elif row["is_dendrite"]:
            return "basal"
        else:
            return "other"
    else:
        if row["is_apical"]:
            if row["dist_to_root"] < proximal_threshold:
                return "prox"
            else:
                return "apical"
        elif row["is_soma"]:
            return "soma"
        elif row["is_dendrite"]:
            if row["dist_to_root"] < proximal_threshold:
                return "prox"
            else:
                return "basal"
        else:
            return "other"


def define_cell_type_comp(row):
    if row["valence"] == "Inh":
        if row["is_soma"]:
            return f'{row["cell_type"]}_soma'
        elif row["is_dendrite"]:
            return f'{row["cell_type"]}_basal'
        else:
            return f'{row["cell_type"]}_other'
    else:
        if row["is_apical"]:
            if row["dist_to_root"] < proximal_threshold:
                return f'{row["cell_type"]}_prox'
            else:
                return f'{row["cell_type"]}_apical'
        elif row["is_soma"]:
            return f'{row["cell_type"]}_soma'
        elif row["is_dendrite"]:
            if row["dist_to_root"] < proximal_threshold:
                return f'{row["cell_type"]}_prox'
            else:
                return f'{row["cell_type"]}_basal'
        else:
            return f'{row["cell_type"]}_other'


def specify_unsure(row):
    if row["cell_type"] == "Unsure":
        if row["classification_system"] == "aibs_coarse_excitatory":
            return "Unsure E"
        else:
            return "Unsure I"
    else:
        return row["cell_type"]


def adjust_layer6(row):
    if row["cell_type"] == "6IT" or row["cell_type"] == "6CT":
        return "6P"
    else:
        return row["cell_type"]


def get_cell_type_df(params):
    if params.get("ALTERNATE_CELL_TYPE") is not None:
        column_df = pd.read_pickle(params.get("ALTERNATE_CELL_TYPE"))
        print(f"Loading cell types from {params.get('ALTERNATE_CELL_TYPE')}")
    else:
        from caveclient import CAVEclient

        client = CAVEclient("minnie65_phase3_v1")

        if params.get("TIMESTAMP") != "none":
            timestamp = datetime.fromtimestamp(float(params.get("TIMESTAMP")))
        else:
            timestamp = None

        mat_version = params.get("MAT_VERSION")
        client.materialize.version = mat_version

        include_classes = ["aibs_coarse_excitatory", "aibs_coarse_inhibitory"]
        column_table = params.get("COLUMN_TABLE", project_info.column_table)
        column_df = client.materialize.query_table(column_table, timestamp=timestamp)
        column_df = column_df.query("classification_system in @include_classes")

    column_df["cell_type"] = column_df.apply(specify_unsure, axis=1)
    ct_merge_df = column_df[["pt_root_id", "cell_type"]]
    ct_merge_df["cell_type"] = ct_merge_df.apply(adjust_layer6, axis=1)
    return ct_merge_df


def _transform_baseline_syn_count(syn_profile_count, bin_column):
    syn_profile_count_long = syn_profile_count.melt(
        value_name="num_syn", ignore_index=False
    ).reset_index()
    syn_profile_count_long = syn_profile_count_long.merge(
        syn_profile_count_long.groupby(bin_column).sum()["num_syn"].rename("bin_total"),
        left_on=bin_column,
        right_index=True,
    )

    syn_profile_count_long["bin_other"] = (
        syn_profile_count_long["bin_total"] - syn_profile_count_long["num_syn"]
    )
    syn_profile_count_long["p"] = (
        syn_profile_count_long["num_syn"] / syn_profile_count_long["bin_total"]
    )
    return syn_profile_count_long


def baseline_counts(syn_df, bin_column, cell_type_column, order):
    syn_profile_count = get_syn_profile_count(
        syn_df,
        bin_column,
        cell_type_column,
        order=order,
    )
    syn_profile_count_long = _transform_baseline_syn_count(
        syn_profile_count, bin_column
    )
    return syn_profile_count, syn_profile_count_long


def get_syn_profile_count(syn_profile_df, bin_column, cell_type_column, order=None):
    syn_profile_count = (
        syn_profile_df.groupby([bin_column, cell_type_column])
        .count()[["id"]]
        .rename(columns={"id": "syn_count"})
        .reset_index()
    )

    syn_profile_count = syn_profile_count.pivot_table(
        index=bin_column, columns=cell_type_column, values="syn_count", fill_value=0
    )

    if order is not None:
        drop_cols = syn_profile_count.columns.values
        drop_cols = drop_cols[~np.isin(drop_cols, order)]
        if len(drop_cols) > 0:
            syn_profile_count = syn_profile_count.drop(columns=drop_cols)

    return syn_profile_count


@click.command()
@click.option("--environment", "-e")
def attach_details_to_baseline(environment):
    params = dotenv.dotenv_values(f"{environment}.env")

    print("Loading config...")
    if params.get("ALTERNATE_E_TYPES") is not None:
        with open(params.get("ALTERNATE_E_TYPES"), "rb") as f:
            e_types = pickle.load(f)
        print("E Types:", e_types)
    else:
        e_types = None

    if params.get("ALTERNATE_I_TYPES") is not None:
        with open(params.get("ALTERNATE_I_TYPES"), "rb") as f:
            i_types = pickle.load(f)
        print("I Types:", i_types)
    else:
        print("Default I types")
        i_types = None

    catConfig = card_config.CategoryConfig(e_types=e_types, i_types=i_types)

    ct_df = get_cell_type_df(params)

    print("Reading baseline...")
    base_environment = params.get("BASE_ENVIRONMENT", environment)
    target_df = pd.read_feather(
        f'{column_file_dir}/baseline_synapses_{base_environment}_{params.get("FILE_TAG")}_post.feather'
    )

    print("Merging cell types...")
    target_df = target_df.merge(
        ct_df.rename(columns={"pt_root_id": "post_pt_root_id"}), on="post_pt_root_id"
    )

    print("Assigning layers...")
    target_df["layer"] = layerConfig.assign_layer(target_df["soma_depth_um"])
    target_df["layer_bin"] = layerConfig.assign_layer_bin(target_df["syn_depth_um"])

    print("Assigning compartments...")
    target_df["comp"] = target_df.apply(define_comp, axis=1)
    target_df["cell_type_comp"] = target_df.apply(define_cell_type_comp, axis=1)

    print("Computing baseline synapse distribution...")

    syn_profile_comp_count, syn_profile_count_comp_long = baseline_counts(
        target_df, "layer_bin", "cell_type_comp", catConfig.order_comp
    )

    baseline_data = {
        "cell_type_df": ct_df,
        "layer_bins": layerConfig.layer_bins,
        "syn_profile_bin_comp_df": target_df,
        "syn_profile_count_comp_long": syn_profile_count_comp_long,
        "syn_profile_comp_count": syn_profile_comp_count,
        "layerConfig": layerConfig,
        "catConfig": catConfig,
    }

    bd_filename = baseline_data_filename(
        f"{project_paths.data}/temp", environment, params.get("FILE_TAG")
    )
    print(f"Saving cell typed baseline to {bd_filename}")
    with open(bd_filename, "wb") as f:
        pickle.dump(baseline_data, f)


if __name__ == "__main__":
    attach_details_to_baseline()
