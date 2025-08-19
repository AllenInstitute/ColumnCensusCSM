import os
import sys
from pathlib import Path
import dotenv

__all__ = ['project_info', 'project_paths']

config_file = '~/Work/Projects/config/minnie_column.env'
params = dotenv.dotenv_values(os.path.expanduser(config_file))
base_dir = os.path.expanduser(params.get('BASE_DIR'))
sys.path.append(f'{base_dir}/src')

import project_setup as ps

project_info = ps.project_info(config_file)
project_paths = ps.project_paths(config_file)
