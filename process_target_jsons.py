import os
import subprocess
import optparse

parser = optparse.OptionParser()

parser.add_option('--targets',
    action="store", dest="targets",
    help="targets", default=None)

options, args = parser.parse_args()

targets = options.targets.split(',')
           
path = "~/pho4/data/pipeline_inputs/gcpbm/"

for target in targets:
    subprocess.run(["caper", "run", "chip.wdl", "-i", path+target+".json"])