import numpy as np
import pandas as pd
import dotenv
import click
from common_setup import project_paths
from affine_transform import minnie_column_transform

tform = minnie_column_transform()

column_file_dir = f"{project_paths.data}/temp"
synapse_file_dir = f"{project_paths.data}/synapse_files"
voxel_resolution = np.array([4, 4, 40])

keep_inds = [
    "id",
    "post_pt_root_id",
    "ctr_pt_position",
    "size",
    "dist_to_root",
    "is_apical",
    "is_soma",
    "is_dendrite",
    "syn_depth_um",
    "soma_depth_um",
    "valence",
]


def define_valence(x):
    if x == "aibs_coarse_excitatory":
        return "Exc"
    elif x == "aibs_coarse_inhibitory":
        return "Inh"
    else:
        return "Uns"


def depth_col(pt_col):
    return tform.apply_project("y", pt_col)


# def soma_depth(pt_position, voxel_resolution):
# return voxel_resolution[1] * pt_position[1] / 1000


@click.command()
@click.option("--environment", "-e")
def assemble_column(environment):
    params = dotenv.dotenv_values(f"{environment}.env")
    column_fn = params.get("BASE_FILE")
    column_df = pd.read_pickle(column_fn)
    column_df["soma_depth"] = depth_col(column_df["pt_position"])

    # column_df["soma_depth"] = column_df["pt_position"].apply(
    # lambda x: soma_depth(x, voxel_resolution)
    # )

    syn_dfs = []
    no_syn_oids = []
    no_syn_files = []
    apical_model = params.get("APICAL_MODEL")
    for oid in column_df["pt_root_id"]:
        try:
            fname = f"{synapse_file_dir}/{apical_model}/{oid}_inputs.feather"
            df = pd.read_feather(fname)
            syn_dfs.append(df)
        except:
            no_syn_oids.append(oid)
            no_syn_files.append(fname)
    all_syn_df = pd.concat(syn_dfs).reset_index(drop=True)
    del syn_dfs
    print(
        f'Baseline file has {len(all_syn_df)} synapses across {len(column_df["pt_root_id"])} cells'
    )

    target_soma_df = column_df[
        ["pt_root_id", "classification_system", "pt_position"]
    ].rename(columns={"pt_root_id": "post_pt_root_id"})
    target_soma_df["soma_depth_um"] = depth_col(target_soma_df["pt_position"])
    # target_soma_df["soma_depth_um"] = target_soma_df["pt_position"].apply(
    # lambda x: voxel_resolution[1] * x[1] / 1000
    # )

    target_soma_df["valence"] = target_soma_df["classification_system"].apply(
        define_valence
    )

    all_syn_df["syn_depth_um"] = depth_col(all_syn_df["ctr_pt_position"])
    # all_syn_df["syn_depth_um"] = all_syn_df["ctr_pt_position"].apply(
    # lambda x: voxel_resolution[1] * x[1] / 1000
    # )
    all_syn_df = all_syn_df.merge(
        target_soma_df[["post_pt_root_id", "valence", "soma_depth_um"]],
        how="left",
        on="post_pt_root_id",
    )

    target_df = all_syn_df[keep_inds]
    target_df_fn = f"{column_file_dir}/baseline_synapses_{environment}_{params.get('FILE_TAG')}_post.feather"

    print(f"Saving baseline file to {target_df_fn}")
    target_df.to_feather(target_df_fn)
    pass


if __name__ == "__main__":
    assemble_column()
