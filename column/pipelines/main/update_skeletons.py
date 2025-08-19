from taskqueue import LocalTaskQueue, TaskQueue
import pandas as pd
import os
import sys
from datetime import datetime
from common_setup import project_paths

sys.path.append(os.path.abspath("../skeletonization/"))
from pcg_skeletonization_tq import create_pcg_skeletonize_tasks

from caveclient import CAVEclient
from dotenv import dotenv_values
import json
import click

# soma_id_manual_radius = {
#     303216: 10_000,
#     302951: 12_000,
#     267033: 10_000,
#     311789: 10_000,
#     347181: 10_000,
# }
with open('skeleton_radius.json', 'r') as f:
    soma_id_manual_radius_str = json.load(f)

soma_id_manual_radius = {}
for k, v in soma_id_manual_radius_str.items():
    soma_id_manual_radius[int(k)] = v

multiaxon_dict = {
    258362: 2,
    307059: 2,
}


def load_radius_dict(df, params):
    radius_dict = {}
    for sid, rad in soma_id_manual_radius.items():
        potential_rid = df.query("soma_id == @sid")["pt_root_id"]
        if len(potential_rid) != 1:
            continue
        else:
            radius_dict[int(potential_rid)] = rad
    return radius_dict


def load_multiaxon_dict(df, params):
    ma_dict = {}
    for sid, val in multiaxon_dict.items():
        potential_rid = df.query("soma_id == @sid")["pt_root_id"]
        if len(potential_rid) != 1:
            continue
        else:
            ma_dict[int(potential_rid)] = val
    return ma_dict


def check_previous_tasks(df, target_path):
    keep_indices = []
    for _, row in df.iterrows():
        try:
            fname = f'{target_path}/{row["pt_root_id"]}.json'
            with open(fname[7:], "r") as f:
                dat = json.load(f)
            keep_indices.append(dat["skeleton_created"] == False)
        except:
            keep_indices.append(False)
    return df.loc[keep_indices]


def skel_exists(oid):
    return os.path.exists(f"{project_paths.skeletons}/skeleton_files/{oid}.h5")


@click.command()
@click.option("--environment", "-e", required=True)
@click.option("--retry/--no-retry", required=False, default=False)
@click.option("--table", "-f", required=False, default=None)
def load_task_queue(environment, retry, table):
    env_path = f"{environment}.env"
    params = dotenv_values(dotenv_path=env_path)
    datastack_name = params.get("DATASTACK")
    if params.get("MAT_VERSION") != "latest":
        materialization_version = int(params.get("MAT_VERSION"))
    else:
        materialization_version = None
    split_threshold = float(params.get("SPLIT_THRESHOLD"))
    target_path = f'file://{params.get("TARGET_PATH_LOCAL")}'
    if params.get("TIMESTAMP") != "none":
        timestamp = datetime.fromtimestamp(float(params.get("TIMESTAMP")))
    else:
        timestamp = None
    df = pd.read_pickle(params.get("BASE_FILE"))

    if retry:
        print("Only retrying failed tasks")
        df = check_previous_tasks(df, target_path=target_path)

    radius_dict = load_radius_dict(df, params)
    ma_dict = load_multiaxon_dict(df, params)

    tq = LocalTaskQueue(parallel=int(params.get("N_PROCESS", 1)))

    task_iter = create_pcg_skeletonize_tasks(
        df.query("skeleton_created == False and ignore_skeleton == False"),
        datastack_name,
        target_path,
        params=params,
        materialization_version=materialization_version,
        split_threshold=split_threshold,
        radius_dict=radius_dict,
        n_axon_dict=ma_dict,
    )
    tq.insert_all(task_iter)
    pass


if __name__ == "__main__":
    load_task_queue()