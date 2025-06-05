from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import flabplatform.flabseg as flabseg


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['Segmentation'] = f'{flabseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
