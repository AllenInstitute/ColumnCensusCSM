import joblib
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import dotenv_values
from meshparty import meshwork

from common_setup import project_info, project_paths
from plotting_code import plot_layers
import tqdm
from loguru import logger

_ = logger.add("apical_classification.log")

import os
import sys

base_path = f"{project_paths.base}/pipelines"
sys.path.append(os.path.abspath(f"{base_path}/apical_classification/src"))
from apical_features import *
from apical_model_utils import *

skel_dir = f"{project_paths.skeletons}/skeleton_files"
preview_dir = f"{project_paths.skeletons}/previews"
synapse_dir = f"{project_paths.data}/synapse_files"
model_dir = os.path.abspath(f"{base_path}/apical_classification/models")

layer_bnds = np.load(f"{project_paths.data}/layer_bounds_v1.npy")
height_bounds = np.load(f"{project_paths.data}/height_bounds_v1.npy")
width_bounds = np.load(f"{project_paths.data}/width_bounds_v1.npy")

rfc = joblib.load(f"{model_dir}/point_model_current.pkl")
feature_cols = joblib.load(f"{model_dir}/feature_cols_current.pkl")
branch_params = joblib.load(f"{model_dir}/branch_params_current.pkl")

BranchClassifier = BranchClassifierFactory(rfc, feature_cols)
branch_classifier = BranchClassifier(**branch_params)


def plot_neuron_with_apical(
    sk,
    dendrite_skind_mask,
    apical_skind_mask,
    input_df,
    oid,
    layer_bnds,
    height_bounds,
    width_bounds,
    cell_type=None,
):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="w", dpi=150)

    axon_mask = ~dendrite_skind_mask
    if np.any(axon_mask):
        ax.scatter(
            x=sk.vertices[axon_mask, 0] / 1000,
            y=sk.vertices[axon_mask, 1] / 1000,
            s=0.2,
            alpha=0.5,
            color=(0.059, 0.780, 1.000),
        )

    basal_mask = np.logical_and(dendrite_skind_mask, ~apical_skind_mask)
    if np.any(basal_mask):
        ax.scatter(
            x=sk.vertices[basal_mask, 0] / 1000,
            y=sk.vertices[basal_mask, 1] / 1000,
            s=0.2,
            alpha=0.5,
            color="k",
        )

    if np.any(apical_skind_mask):
        ax.scatter(
            x=sk.vertices[apical_skind_mask, 0] / 1000,
            y=sk.vertices[apical_skind_mask, 1] / 1000,
            s=0.2,
            alpha=0.5,
            color="r",
        )

    ax.plot(
        sk.vertices[sk.root, 0] / 1000,
        sk.vertices[sk.root, 1] / 1000,
        marker="o",
        color="w",
        markersize=5,
        markeredgecolor="k",
    )

    ax.set_aspect("equal")
    plot_layers(
        layer_bnds,
        height_bounds,
        width_bounds,
        ax=ax,
        linestyle=":",
        linewidth=1,
        color="k",
    )
    ax.set_title(
        f'{oid} | {cell_type} | {len(input_df.query("is_apical"))}/{len(input_df.query("is_soma"))}/{len(input_df)}'
    )
    return fig


@click.command()
@click.option("--environment", "-e", required=True)
def run_update_synapse_files(environment):
    env_path = f"{environment}.env"
    params = dotenv_values(dotenv_path=env_path)

    model_name = params.get("APICAL_MODEL", "current")
    synapse_model_dir = f"{synapse_dir}/{model_name}"
    if not os.path.exists(synapse_model_dir):
        os.makedirs(synapse_model_dir)
    preview_model_dir = f"{preview_dir}/{model_name}"
    if not os.path.exists(preview_model_dir):
        os.makedirs(preview_model_dir)

    column_df_fn = params.get("BASE_FILE")
    column_df = pd.read_pickle(column_df_fn)

    for _, row in tqdm.tqdm(column_df.query("ignore_skeleton==False").iterrows()):
        oid = row["pt_root_id"]
        if not os.path.exists(f"{skel_dir}/{oid}.h5"):
            logger.error(f"No file for {oid}")
            continue
        if os.path.exists(f"{synapse_model_dir}/{oid}_inputs.feather"):
            continue
        try:
            skel_file = f"{skel_dir}/{oid}.h5"
            nrn = meshwork.load_meshwork(skel_file)
            sk = nrn.skeleton
            if row["classification_system"] == "aibs_coarse_excitatory":
                point_features_df = process_apical_features(nrn, peel_threshold=0.1)

                dendrite_synapse_indices = nrn.anno.post_syn.df.index
                dendrite_skind_mask = nrn.skeleton.node_mask
                nrn.reset_mask()

                input_df = nrn.anno.post_syn.df.reset_index(drop=True)
                branch_df = branch_classifier.fit_predict_data(
                    point_features_df, "base_skind"
                )
                apical_base = branch_df.query("is_apical")["base_skind"].values

                if len(apical_base) > 0:

                    sk_downstream = nrn.skeleton.downstream_nodes(apical_base)
                    apical_sk_mask = (
                        np.sum(
                            np.vstack([x.to_skel_mask for x in sk_downstream]), axis=0
                        )
                        > 0
                    )
                    apical_mesh_mask = (
                        np.sum(
                            np.vstack([x.to_mesh_mask for x in sk_downstream]), axis=0
                        )
                        > 0
                    )

                    apical_post_inds = nrn.anno.post_syn.filter_query(
                        apical_mesh_mask
                    ).df.index
                    input_df.loc[apical_post_inds, "is_apical"] = True
                    input_df["is_apical"] = input_df["is_apical"].fillna(False)
                else:
                    input_df["is_apical"] = False
                    apical_sk_mask = np.full(len(sk.vertices), False)
            else:
                nrn = apply_dendrite_mask(nrn)
                dendrite_synapse_indices = nrn.anno.post_syn.df.index
                dendrite_skind_mask = sk.node_mask

                nrn.reset_mask()

                input_df = nrn.anno.post_syn.df.reset_index(drop=True)
                input_df["is_apical"] = False
                apical_sk_mask = np.full(len(sk.vertices), False)

            soma_post_inds = nrn.anno.post_syn.filter_query(
                nrn.root_region.to_mesh_mask
            ).df.index
            input_df.loc[
                np.intersect1d(soma_post_inds, input_df.index), "is_soma"
            ] = True
            input_df["is_soma"] = input_df["is_soma"].fillna(False)

            input_df.loc[
                np.intersect1d(dendrite_synapse_indices, input_df.index), "is_dendrite"
            ] = True
            input_df["is_dendrite"] = input_df["is_dendrite"].fillna(False)

            input_df["dist_to_root"] = (
                nrn.distance_to_root(input_df["post_pt_mesh_ind"]) / 1000
            )
            input_df.to_feather(f"{synapse_model_dir}/{oid}_inputs.feather")
            logger.info(f"Saved synapses for {oid}")

            fig = plot_neuron_with_apical(
                sk,
                dendrite_skind_mask,
                apical_sk_mask,
                input_df,
                oid,
                layer_bnds,
                height_bounds,
                width_bounds,
            )
            fig.savefig(
                f"{preview_model_dir}/apical_preview_{oid}.png", bbox_inches="tight"
            )
            logger.info(f"Saved preview for {oid}")
            plt.close(fig)
        except Exception as e:
            logger.error(f"{oid} | {e}")


if __name__ == "__main__":
    run_update_synapse_files()