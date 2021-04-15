import os
import subprocess
import optparse

parser = optparse.OptionParser()

parser.add_option('--targets',
    action="store", dest="targets",
    help="targets", default=None)

options, args = parser.parse_args()

targets = options.targets.split(',')

for target in targets:
    subprocess.run(["bash", "/users/amr1/pho4/postprocess_target.sh", target])