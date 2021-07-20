import os
import subprocess
import optparse

parser = optparse.OptionParser()
parser.add_option('--targets',
    action="store", dest="targets",
    help="targets", default=None)
parser.add_option('--gpus',
    action="store", dest="gpus",
    help="gpus", default=None)
options, args = parser.parse_args()

targets = options.targets.split(',')

for target in targets:
    subprocess.run(["python", "train_noCtl.py", "--target", target, "--gpus", options.gpus])