import pandas as pd
import numpy as np
from datetime import datetime
from caveclient import CAVEclient
from common_setup import project_paths, project_info
import click
from utils.skel_env_template import env_file_template
from utils.skel_db import skeletons_created, skeletons_ignored

include_categories = ["aibs_coarse_excitatory", "aibs_coarse_inhibitory"]
soma_include_categories = ["neuron"]


def add_num_soma(df, client, soma_table, timestamp=None):
    soma_df = client.materialize.query_table(
        soma_table,
        filter_in_dict={"pt_root_id": np.unique(df["pt_root_id"])},
        timestamp=timestamp,
    )
    soma_df = soma_df.query("cell_type in @soma_include_categories").reset_index(
        drop=True
    )
    soma_df["num_soma"] = soma_df.groupby("pt_root_id").transform("count")["id"]
    soma_df = soma_df.rename(columns={"id": "soma_id"})

    soma_df.drop_duplicates(subset="pt_root_id", inplace=True)
    return df.merge(soma_df[["pt_root_id", "num_soma", "soma_id"]], how="left").fillna(
        0
    )


@click.command()
@click.option("--environment", "-e", required=True)
@click.option("--column-table", "-c", required=False, default=project_info.column_table)
@click.option("--apical-model", "-a", required=False, default="v1")
@click.option("--now/--no-now", default=False)
@click.option("--version", "-v", required=False, default=None)
def update_column_dataframe(environment, column_table, apical_model, now, version):
    datastack = project_info.datastack
    client = CAVEclient(datastack)
    if version is not None:
        client.materialize.version = version

    if now:
        timestamp = datetime.now()
    else:
        timestamp = None
    column_df = client.materialize.query_table(column_table, timestamp=timestamp)
    column_df = column_df.query("classification_system in @include_categories")

    column_df = add_num_soma(
        column_df, client, project_info.soma_table, timestamp=timestamp
    )
    column_df = column_df.query("num_soma <= 1").reset_index(drop=True)

    column_df["skeleton_created"] = skeletons_created(column_df["pt_root_id"])
    column_df["ignore_skeleton"] = skeletons_ignored(column_df["pt_root_id"])

    if timestamp is None:
        file_tag = f"v{client.materialize.version}"
    else:
        file_tag = f"t{int(timestamp.timestamp())}"

    column_df.to_pickle(f"{project_paths.data}/temp/{environment}_{file_tag}_soma.pkl")
    if timestamp is None:
        print(f"Saved new file for {environment} v{client.materialize.version}")
    else:
        print(f"Saved new file for {environment} {timestamp}")

    print(
        f"{len(column_df.query('skeleton_created == False and ignore_skeleton == False'))} skeletons to generate"
    )

    env_file_template(
        f"{environment}.env",
        environment,
        client.materialize.version,
        column_table,
        apical_model,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    update_column_dataframe()
