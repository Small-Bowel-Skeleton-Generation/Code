import ocnn

import datasets
import models


def get_dataset(flags):
  if flags.name.lower() == 'shapenet':
    return datasets.dualoctree_snet.get_shapenet_dataset(flags)
  else:
    raise ValueError
