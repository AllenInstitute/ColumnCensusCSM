import os
from collections import namedtuple

import dotenv

__all__ = ['project_paths', 'project_info']


def project_paths(config_file):
    params = dotenv.dotenv_values(os.path.expanduser(config_file))

    base_dir = os.path.expanduser(params.get('BASE_DIR'))
    dir_names = ['base', 'data', 'notebooks', 'plots', 'skeletons', 'src']
    DefaultDirs = namedtuple('DefaultDirs', dir_names)

    project_paths = DefaultDirs(base_dir,
                                f'{base_dir}/data',
                                f'{base_dir}/notebooks',
                                f'{base_dir}/plots',
                                f'{base_dir}/skeletons',
                                f'{base_dir}/src')
    return project_paths


def project_info(config_file):
    info_params = dotenv.dotenv_values(os.path.expanduser(config_file))

    info_names = ['datastack', 'soma_table', 'synapse_table',
                  'column_table', 'proofreading_table', 'slant_table']
    DefaultInfo = namedtuple('DefaultInfo', info_names)

    project_info = DefaultInfo(info_params.get('DATASTACK_NAME'),
                               info_params.get('SOMA_TABLE'),
                               info_params.get('SYNAPSE_TABLE'),
                               info_params.get('COLUMN_TABLE'),
                               info_params.get('PROOFREADING_TABLE'),
                               info_params.get('SLANT_TABLE'))
    return project_info
