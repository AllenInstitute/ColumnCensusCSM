import dataset
import numpy as np
import pandas as pd
import os

base_file = os.path.dirname(os.path.abspath(__file__))
db_name = f"{base_file}/skel_db/skel_database.db"
skel_table_name = "skeleton_creation"

"""
Simple data schema -
    * root_id : Skeleton root id
    * skeleton_created : Whether a skeleton has already been created
    * skip_skeleton : If True, no attempt will be made to skeletonize
    * version : Materialization version skeleton came from
"""
db = dataset.connect(f"sqlite:///{db_name}")


def parse_query(records):
    records = list(records)
    if len(records) == 0:
        return None
    else:
        return pd.DataFrame.from_records(records)


def skeletons_created(root_ids):
    if not isinstance(root_ids, list):
        try:
            root_ids = root_ids.tolist()
        except:
            raise ValueError("Root IDs must be a list or have a .tolist() method")

    skel_table = db[skel_table_name]
    df = parse_query(skel_table.find(root_id=root_ids))
    if df is None:
        existing_ids = []
    else:
        existing_ids = df.query("skip_skeleton == False and skeleton_created == True")[
            "root_id"
        ].values
    return np.isin(root_ids, np.unique(existing_ids))


def skeletons_ignored(root_ids):
    if not isinstance(root_ids, list):
        try:
            root_ids = root_ids.tolist()
        except:
            raise ValueError("Root IDs must be a list or have a .tolist() method")

    skel_table = db[skel_table_name]
    df = parse_query(skel_table.find(root_id=root_ids))
    if df is None:
        existing_ids = []
    else:
        existing_ids = df.query("skip_skeleton == True")["root_id"].values
    return np.isin(root_ids, np.unique(existing_ids))


def add_finished_skeletons(root_ids, version, created=True):
    if not pd.api.types.is_list_like(created):
        created = len(root_ids) * [created]

    rows = [
        {
            "root_id": root_id,
            "skeleton_created": success,
            "skip_skeleton": False,
            "version": version,
        }
        for root_id, success in zip(root_ids, created)
    ]

    skel_table = db[skel_table_name]
    skel_table.insert_many(rows)
    return len(rows)


def reset_skeleton_created(root_ids):
    if not isinstance(root_ids, list):
        try:
            root_ids = root_ids.tolist()
        except:
            raise ValueError("Root IDs must be a list or have a .tolist() method")

    skel_table = db[skel_table_name]
    df = parse_query(skel_table.find(root_id=root_ids))
    df['skeleton_created'] = False
    for _, record in df.to_dict(orient='index').items():
        skel_table.update(record, 'id')
    return