import card_config
import warnings
from common_setup import project_info, project_paths
from caveclient import CAVEclient
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bitsandbobs as bab
import tqdm

from src.loading_core_data import *
from src.synapse_budget import *
from src.card_plotting_code import *
from src.neuron_info import *

voxel_resolution = np.array([4, 4, 40])
data_dir = project_paths.data
process_dir = "synapse_files"

synapse_table = project_info.synapse_table
ct_column_table = project_info.column_table
soma_table = project_info.soma_table
proofreading_table = project_info.soma_table

warnings.simplefilter("ignore")

# Initialize configuration options

visConfig = card_config.VisualizationConfig()
layerConfig = card_config.LayerConfig()
catConfig = card_config.CategoryConfig()

simplifyLayer6Remap = card_config.SimplifyLayer6Remap(catConfig)
basicLayer6Remap = card_config.BasicLayer6Remap()

if __name__ == "__main__":
    client = CAVEclient(project_info.datastack)
    print(f"Making cards for {client.materialize.version}")

    plot_dir = bab.assert_dir(
        f"{project_paths.plots}/cards/mat_v{client.materialize.version}"
    )

    # Grab data from the database
    soma_df, soma_count_df = get_soma_data(soma_table, client)
    column_df, order = get_cell_type_data(ct_column_table, soma_df, client)

    # Sort out bin labels
    layer_bins = generate_profile_bins(
        layerConfig.layer_bounds, layerConfig.height_bounds, layerConfig.bins_per_layer
    )

    # Loading synapses
    try:
        print(f"Loading synapses for version {client.materialize.version}")
        f = f"{process_dir}/column_synapses_v{client.materialize.version}_post.feather"
        syn_profile_df = pd.read_feather(f)
    except:
        raise Exception(
            f'No column synapses for version {client.materialize.version}.\nRun "python update_column_synapses.py --valence post" in terminal.'
        )

    # Compartment labels

    fn = f"{process_dir}/syn_profile_comp_v{client.materialize.version}.feather"
    if not os.path.exists(fn):
        print("\tGenerating compartment labels")
        syn_profile_ct_df = cell_typed_synapse_profile(syn_profile_df, column_df)
        syn_profile_comp_df = synapse_compartment_profile(syn_profile_df, column_df)

        # Apply layer column and rename layer 6 types
        syn_profile_comp_df = simplifyLayer6Remap.remap(syn_profile_bin_comp_df)

        column_df = basicLayer6Remap.remap(column_df)

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

    bin_comp_fn = (
        f"{process_dir}/syn_profile_bin_comp_v{client.materialize.version}.feather"
    )
    comp_count_fn = (
        f"{process_dir}/syn_profile_comp_count_v{client.materialize.version}.feather"
    )
    comp_count_long_fn = f"{process_dir}/syn_profile_comp_count_long_v{client.materialize.version}.feather"
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

    # Generate cards
    print("\tGenerating cards!")
    inh_oids = column_df.query('valence=="Inh"')["pt_root_id"].values
    failed_oids = []
    for oid in tqdm.tqdm(inh_oids):
        try:
            fig = neuron_card(
                oid,
                synapse_table=synapse_table,
                client=client,
                soma_df=column_df,
                column_df=column_df,
                layerConfig=layerConfig,
                visConfig=visConfig,
                catConfig=catConfig,
                layer_bins=layer_bins,
                background_comp_long=syn_profile_count_comp_long,
                soma_count_df=soma_count_df,
                cell_type_column="cell_type_comp",
                save_data=False,
                data_dir=process_dir,
            )

            nrn_subtype, nrn_soma_id, soma_position = neuron_info(oid, column_df)
            fig.savefig(
                f"{plot_dir}/neuron_card_v2_{nrn_subtype}_sid{nrn_soma_id}_oid{oid}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            print(e)
            failed_oids.append(oid)
            continue
