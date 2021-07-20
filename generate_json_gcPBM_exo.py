import os
import json
import subprocess
import optparse

parser = optparse.OptionParser()

parser.add_option('--targets',
    action="store", dest="targets",
    help="targets", default=None)

options, args = parser.parse_args()

targets = options.targets.split(',')
           
path = "/oak/stanford/groups/akundaje/amr1/pho4/raw/gcpbm_chipexo/"
dest = "data/pipeline_inputs/gcpbm/"

for target in targets:
    jsonDict = {}
    jsonDict["chip.pipeline_type"] = "tf"
    jsonDict["chip.title"] = target
    jsonDict["chip.description"] = f"chipexo of {target}"
    jsonDict["chip.enable_count_signal_track"] = True
    jsonDict["chip.peak_caller"] = "macs2"
    jsonDict["chip.genome_tsv"] = "/mnt/data/pipeline_genome_data/genome_tsv/v2/hg38_klab.tsv"

    jsonDict["chip.paired_end"] = True
    for filename in os.listdir(path+target):
        if filename.endswith('.fastq.gz') == False:
            jsonDict[f"chip.fastqs_rep1_R1"] = [f"{path}{target}/{filename}_1.fastq.gz"]
            jsonDict[f"chip.fastqs_rep1_R2"] = [f"{path}{target}/{filename}_2.fastq.gz"]
            break

    with open(f'{dest}{target}.json', 'w') as f:
            json.dump(jsonDict, f)