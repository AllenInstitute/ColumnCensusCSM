import pandas as pd
import os
import shutil
import glob
import click
from pcg_skel import chunk_cache
from dotenv import dotenv_values
from meshparty import meshwork
import tqdm
import json

from utils.skel_db import add_finished_skeletons


def update_cache_from_neuron(nrn, cache_fn):
    l2ids_all = nrn.anno.lvl2_ids.df["lvl2_id"].values
    chunk_cache.save_ids_to_cache(
        l2ids_all[nrn.skeleton.mesh_index], nrn.skeleton.vertices, cache_fn
    )


def update_l2cache(column_name, cache_fn, pickle_fn, skel_cf_path):
    df = pd.read_pickle(pickle_fn)
    root_ids = df.query("skeleton_created == False")[column_name].values
    for root_id in tqdm.tqdm(root_ids):
        nrn_fn = f"{skel_cf_path}/{root_id}.h5"
        try:
            nrn = meshwork.load_meshwork(nrn_fn)
        except:
            continue
        update_cache_from_neuron(nrn, cache_fn)
    pass


def clean_directory(target_directory):
    json_files = sorted(glob.glob(f"{target_directory}/*.json"))
    preview_files = sorted(glob.glob(f"{target_directory}/*_preview.png"))
    skeleton_files = sorted(glob.glob(f"{target_directory}/*.h5"))

    json_targ = f"{target_directory}/json_files"
    preview_targ = f"{target_directory}/previews"
    skeleton_targ = f"{target_directory}/skeleton_files"

    skinds_done = []
    for f in json_files:
        with open(f, "r") as jsonf:
            dat = json.load(jsonf)
            if dat["skeleton_created"] == False:
                continue
            skinds_done.append(dat["root_id"])
        shutil.move(f, f"{json_targ}/{os.path.split(f)[1]}")

    r = add_finished_skeletons(skinds_done, version=1, created=True)
    print(f"Adding {r} neurons to finished skeletons")

    for f in preview_files:
        shutil.move(f, f"{preview_targ}/{os.path.split(f)[1]}")

    for f in skeleton_files:
        shutil.move(f, f"{skeleton_targ}/{os.path.split(f)[1]}")
    pass


@click.command()
@click.option("--environment", "-e", required=True)
def clean_and_cache(environment):
    env_path = f"{environment}.env"
    params = dotenv_values(dotenv_path=env_path)
    target_directory = params.get("TARGET_PATH_LOCAL")
    print(f"Working on directory {target_directory}")
    print("Updating L2 cache")

    cache_fn = params.get("CACHE_FILE")
    pickle_fn = params.get("BASE_FILE")
    skel_cf_path = params.get("TARGET_PATH_LOCAL")

    # update_l2cache("pt_root_id", cache_fn, pickle_fn, skel_cf_path)
    print("Cleaning Directory")
    clean_directory(target_directory)

if __name__ == "__main__":
    clean_and_cache()