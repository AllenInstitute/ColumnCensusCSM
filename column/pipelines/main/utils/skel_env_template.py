from common_setup import project_info, project_paths


def env_file_template(
    filename, environment, mat_version, column_table, apical_model, timestamp=None
):
    if timestamp is None:
        template_file = f"""
    PICKLE_FILE={project_paths.data}/temp/new_{environment}_v{mat_version}_soma.pkl
    BASE_FILE={project_paths.data}/temp/{environment}_v{mat_version}_soma.pkl
    MAT_VERSION={mat_version}
    FILE_TAG=v{mat_version}
    CACHE_FILE={project_paths.data}/l2lookup.sqlite
    TARGET_PATH_LOCAL={project_paths.skeletons}
    TARGET_PATH_GS=gs://allen-minnie-phase3/minniephase3-pcg-skeletons
    QUEUE_URL=https://sqs.us-east-2.amazonaws.com/629034007606/pcg_skel
    N_PROCESS=8
    REFINE_SKELETON=True
    SPLIT_THRESHOLD=0.7
    KEEP_SYNAPSES=True
    SAVE_PREVIEW=True
    INVALIDATION_D=3
    APICAL_MODEL={apical_model}
    SYNAPSE_TABLE={project_info.synapse_table}
    COLUMN_TABLE={column_table}
    DATASTACK={project_info.datastack}
    LAYER_BOUNDS={project_paths.data}/layer_bounds_v1.npy
    HEIGHT_BOUNDS={project_paths.data}/height_bounds_v1.npy
    WIDTH_BOUNDS={project_paths.data}/width_bounds_v1.npy
    TIMESTAMP=none
    """
    else:
        template_file = f"""
    PICKLE_FILE={project_paths.data}/temp/new_{environment}_t{int(timestamp.timestamp())}_soma.pkl
    BASE_FILE={project_paths.data}/temp/{environment}_t{int(timestamp.timestamp())}_soma.pkl
    MAT_VERSION={mat_version}
    FILE_TAG=t{int(timestamp.timestamp())}
    CACHE_FILE={project_paths.data}/l2lookup.sqlite
    TARGET_PATH_LOCAL={project_paths.skeletons}
    TARGET_PATH_GS=gs://allen-minnie-phase3/minniephase3-pcg-skeletons
    QUEUE_URL=https://sqs.us-east-2.amazonaws.com/629034007606/pcg_skel
    N_PROCESS=8
    REFINE_SKELETON=True
    SPLIT_THRESHOLD=0.7
    KEEP_SYNAPSES=True
    SAVE_PREVIEW=True
    INVALIDATION_D=3
    APICAL_MODEL={apical_model}
    SYNAPSE_TABLE={project_info.synapse_table}
    COLUMN_TABLE={column_table}
    DATASTACK={project_info.datastack}
    LAYER_BOUNDS={project_paths.data}/layer_bounds_v1.npy
    HEIGHT_BOUNDS={project_paths.data}/height_bounds_v1.npy
    WIDTH_BOUNDS={project_paths.data}/width_bounds_v1.npy
    TIMESTAMP={timestamp.timestamp()}
    """

    with open(filename, "w") as f:
        f.write(template_file)