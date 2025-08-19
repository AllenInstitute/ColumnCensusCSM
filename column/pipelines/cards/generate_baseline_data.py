import json
from re import A
import card_config
import warnings
from caveclient import CAVEclient
import os
import numpy as np
import pandas as pd
import pickle
import click
import dotenv

from src.loading_core_data import *
from src.synapse_budget import *
from src.card_plotting_code import *
from src.neuron_info import *


def baseline_data_filename(process_dir, configuration_name, version):
    return f"{process_dir}/{configuration_name}_v{version}_files.pkl"


configuration_name = "simple_types"

voxel_resolution = np.array([4, 4, 40])
process_dir = "synapse_files"

warnings.simplefilter("ignore")

# Initialize configuration options

visConfig = card_config.VisualizationConfig()
layerConfig = card_config.LayerConfig()
catConfig = card_config.CategoryConfig()

simplifyLayer6Remap = card_config.SimplifyLayer6Remap(catConfig)
basicLayer6Remap = card_config.BasicLayer6Remap()


@click.command()
@click.option("-c", "--config", required=False, default=None)
@click.option("-v", "--version", required=False, default=None, type=int)
@click.option("-s", "--soma_table", required=False, default=None)
@click.option("-t", "--cell_table", required=False, default=None)
@click.option("-d", "--datastack", required=False, default="minnie65_phase3_v1")
def cli(config, version, soma_table, cell_table, datastack):
    if config is not None:
        vals = dotenv.dotenv_values(config)
        datastack = vals.get("DATASTACK", datastack)
        cell_table = vals.get("CELL_TYPE_TABLE", cell_table)
        soma_table = vals.get("SOMA_TABLE", soma_table)

    client = CAVEclient(datastack)
    if version is not None:
        client.materialize.version = version
    print(f"Making cards for v{client.materialize.version}")

    # Grab data from the database
    soma_df = client.materialize.query_table(soma_table)
    column_df = client.materialize.query_table(cell_table)

    soma_df = soma_df.drop(soma_df[soma_df["pt_root_id"] == 0].index)
    soma_count_df = (
        soma_df[["pt_root_id", "id"]]
        .groupby("pt_root_id")
        .count()
        .reset_index()
        .rename(columns={"id": "num_soma"})
    )
    soma_df = soma_df.merge(soma_count_df.query("num_soma == 1"), on="pt_root_id").drop(
        columns="num_soma"
    )

    # Sort out bin labels
    layer_bins = generate_profile_bins(
        layerConfig.layer_bounds, layerConfig.height_bounds, layerConfig.bins_per_layer
    )

    # Loading synapses
    try:
        print(f"Loading synapses for version {client.materialize.version}")
        f = f"{process_dir}/baseline_synapses_v{client.materialize.version}_post.feather"
        print(f)
        syn_profile_df = pd.read_feather(f)
    except:
        raise Exception(
            f'No column synapses for version {client.materialize.version}.\nRun "python update_column_synapses.py --valence post" in terminal.'
        )

    # Compartment label
    fn = f"{process_dir}/{configuration_name}_syn_profile_comp_v{client.materialize.version}.feather"
    if not os.path.exists(fn):
        print("\tGenerating compartment labels")
        syn_profile_ct_df = cell_typed_synapse_profile(syn_profile_df, column_df)
        syn_profile_comp_df = synapse_compartment_profile(syn_profile_ct_df, column_df)

        # Apply layer column and rename layer 6 types
        syn_profile_comp_df = simplifyLayer6Remap.remap(syn_profile_comp_df)
        column_df = basicLayer6Remap.remap(column_df)

        ##
        syn_profile_comp_df["cell_type_comp"] = add_layer_column(
            syn_profile_comp_df,
            "cell_type_comp",
            column_df,
            "soma_depth_um",
            layerConfig.layer_bounds,
            layerConfig.layers,
            catConfig.i_types,
        )

        syn_profile_comp_df.to_feather(fn)
    else:
        syn_profile_comp_df = pd.read_feather(fn)

    # Run baseline counts
    bin_comp_fn = f"{process_dir}/{configuration_name}_syn_profile_bin_comp_v{client.materialize.version}.feather"
    comp_count_fn = f"{process_dir}/{configuration_name}_syn_profile_comp_count_v{client.materialize.version}.feather"
    comp_count_long_fn = f"{process_dir}/{configuration_name}_syn_profile_comp_count_long_v{client.materialize.version}.feather"
    if (
        os.path.exists(bin_comp_fn)
        and os.path.exists(comp_count_fn)
        and os.path.exists(comp_count_long_fn)
    ):
        syn_profile_bin_comp_df = pd.read_feather(bin_comp_fn)
        syn_profile_comp_count = pd.read_feather(comp_count_fn)
        syn_profile_count_comp_long = pd.read_feather(comp_count_long_fn)
    else:
        print("\tGenerating baseline counts")
        (
            syn_profile_bin_comp_df,
            syn_profile_comp_count,
            syn_profile_count_comp_long,
        ) = baseline_counts(
            syn_profile_comp_df,
            "cell_type_comp",
            layer_bins,
            catConfig.order_comp,
        )
        syn_profile_bin_comp_df.to_feather(bin_comp_fn)
        syn_profile_comp_count.to_feather(comp_count_fn)
        syn_profile_count_comp_long.to_feather(comp_count_long_fn)

    baseline_data = {
        "soma_df": soma_df,
        "soma_count_df": soma_count_df,
        "cell_type_df": column_df,
        "layer_bins": layer_bins,
        "syn_profile_bin_comp_df": syn_profile_bin_comp_df,
        "syn_profile_count_comp_long": syn_profile_count_comp_long,
        "syn_profile_comp_count": syn_profile_comp_count,
        "layerConfig": layerConfig,
        "catConfig": catConfig,
    }
    bd_filename = baseline_data_filename(
        process_dir, configuration_name, client.materialize.version
    )
    with open(bd_filename, "w") as f:
        pickle.dump(baseline_data, f)
    print(f"\tSaved baseline data to {bd_filename}")
    pass


if __name__ == "__main__":
    cli()
