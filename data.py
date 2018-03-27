from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import random

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, dest='data_dir', default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 45000,
    'valid': 5000,
    'test': 10000,
}


def record_dataset(data_dir):
  filenames = get_filenames(data_dir)

  label_bytes = 1
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = label_bytes + image_bytes
  contents = []
  for filename in filenames:
    with open(filename, 'rb') as f:
      content = f.read(record_bytes)
      while content:
        assert len(content) == record_bytes
        contents.append(content)
        content = f.read(record_bytes)
  assert len(contents) == _NUM_IMAGES['train'] + _NUM_IMAGES['valid']
  valid_index = random.sample(range(len(contents)), _NUM_IMAGES['valid'])
  valid_data = []
  train_data = []
  for i in range(len(contents)):
    if i in valid_index:
      valid_data.append(contents[i])
    else:
      train_data.append(contents[i])
  assert len(train_data) == _NUM_IMAGES['train']
  assert len(valid_data) == _NUM_IMAGES['valid']

  num_train_data = len(train_data)
  num_train_data_per_file = num_train_data // _NUM_DATA_FILES
  for i in range(1, _NUM_DATA_FILES+1):
    with open(os.path.join(data_dir, 'train_batch_%d.bin' % i), 'wb') as f:
      for j in range((i-1)*num_train_data_per_file, i*num_train_data_per_file):
        f.write(train_data[j])

  with open(os.path.join(data_dir, 'valid_batch.bin'), 'wb') as f:
    for c in valid_data:
      f.write(c)
  

def get_filenames(data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  return [
      os.path.join(data_dir, 'data_batch_%d.bin' % i)
      for i in range(1, _NUM_DATA_FILES + 1)
  ]


def main(args):
  record_dataset(args.data_dir)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
