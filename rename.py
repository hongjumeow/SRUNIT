import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--scene', type=str, default='Scene01')
parser.add_argument('--dataroot', type=str, default='.')

opt = parser.parse_args()
file = os.listdir(opt.dataroot)

for f in file:
    source = opt.dataroot + '/' + f
    target = source.replace('rgb', opt.scene)
    os.rename(source, target)

