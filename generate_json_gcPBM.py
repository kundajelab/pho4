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

# targets = ["runx1_cd34", "runx1_k562_1", "runx1_k562_2", "runx2_nho",
#            "runx2_msc_undiff", "runx2_msc_diff", "e2f3_mk562", "e2f1_hela",
#            "e2f1_mmcf7", "e2f1_mhepg2", "e2f1_k562", "e2f4_hela",
#            "e2f4_mhepg2", "e2f4_mmcf7", "e2f4_gm12878", "e2f4_k562",
#            "ets1_mhepg2", "ets1_k562", "ets1_gm12878", "ets1_hepg2",
#            "ets1_gm23338", "elk1_hela", "elk1_mhepg2", "elk1_mcf7",
#            "elk1_imr90", "elk1_a549", "elk1_gm12878", "elk1_k562",
#            "gabpa_gm12878", "gabpa_hepg2", "gabpa_mhepg2", "gabpa_mk562",
#            "gabpa_mmcf7", "gabpa_l32", "gabpa_l4", "gabpa_mcf7",
#            "gabpa_eth_a549", "gabpa_h1", "gabpa_hela", "gabpa_hl60",
#            "gabpa_sk", "gabpa_k562", "mxi_hela", "mxi_k562",
#            "mxi_h1", "mxi_hepg2", "mxi_sk", "mxi_imr90",
#            "mxi_gm12878", "mxi_neural", "myc_k562_1", "myc_k562_2",
#            "myc_k562_3", "myc_hela_1", "myc_hela_2", "myc_nb4",
#            "myc_gm12878", "myc_hepg2", "myc_a549", "myc_h1_1",
#            "myc_h1_2", "myc_mmcf10_4", "myc_mmcf10_36", "myc_mcf7",
#            "myc_mcf7_estrogen", "myc_mcf7_stimu", "myc_mcf7_starv", "myc_huvec",
#            "myc_k562_a05", "myc_k562_a6", "myc_k562_g05", "myc_k562_g6",
#            "max_k562_1", "max_k562_2", "max_k562_3", "max_hela_1",
#            "max_hela_2", "max_mhepg2", "max_hepg2_1", "max_hepg2_2",
#            "max_gm12878", "max_l32", "max_l4", "max_huvec",
#            "max_hct", "max_h1_1", "max_h1_2", "max_nb4",
#            "max_a549_1", "max_a549_2", "max_mcf7", "max_ishikawa",
#            "max_sk"]
           
path = "/oak/stanford/groups/akundaje/amr1/pho4/raw/gcpbm/"
dest = "data/pipeline_inputs/gcpbm/"

for target in targets:
    jsonDict = {}
    jsonDict["chip.pipeline_type"] = "tf"
    jsonDict["chip.title"] = target
    jsonDict["chip.description"] = f"chipseq of {target}"
    jsonDict["chip.enable_count_signal_track"] = True
    jsonDict["chip.always_use_pooled_ctl"] = True
    jsonDict["chip.peak_caller"] = "macs2"
    jsonDict["chip.genome_tsv"] = "/mnt/data/pipeline_genome_data/genome_tsv/v2/hg38_klab.tsv"

    paired_endedness =  False
    for filename in os.listdir(path+target):
        if filename == 'ctl': continue
        if filename.endswith("_2.fastq.gz"):
            paired_endedness = True
            break
    if paired_endedness:
        jsonDict["chip.paired_end"] = True
        keys = []
        for filename in os.listdir(path+target):
            if filename == 'ctl' or filename.endswith('.fastq.gz') == False: continue
            key = filename.split('_')[0]
            if key not in keys: keys.append(key)
        for rep, key in enumerate(keys):
            jsonDict[f"chip.fastqs_rep{rep+1}_R1"] = [f"{path}{target}/{key}_1.fastq.gz"]
            jsonDict[f"chip.fastqs_rep{rep+1}_R2"] = [f"{path}{target}/{key}_2.fastq.gz"]
    else:
        jsonDict["chip.paired_end"] = False
        jsonDict["chip.no_dup_removal"] = True
        rep = 1
        for filename in os.listdir(path+target):
            if filename == 'ctl' or filename.endswith('.fastq.gz') == False: continue
            jsonDict[f"chip.fastqs_rep{rep}_R1"] = [f"{path}{target}/{filename}"]
            rep += 1

    if os.path.exists(path+target+"/ctl"):
        ctl_paired_endedness =  False
        for filename in os.listdir(path+target+"/ctl"):
            if filename.endswith("_2.fastq.gz"):
                ctl_paired_endedness = True
                break
        if ctl_paired_endedness:
            jsonDict["chip.ctl_paired_end"] = True
            keys = []
            for filename in os.listdir(path+target+"/ctl"):
                if filename.endswith('.fastq.gz') == False: continue
                key = filename.split('_')[0]
                if key not in keys: keys.append(key)
            for rep, key in enumerate(keys):
                jsonDict[f"chip.ctl_fastqs_rep{rep+1}_R1"] = [f"{path}{target}/ctl/{key}_1.fastq.gz"]
                jsonDict[f"chip.ctl_fastqs_rep{rep+1}_R2"] = [f"{path}{target}/ctl/{key}_2.fastq.gz"]
        else:
            jsonDict["chip.ctl_paired_end"] = False
            rep = 1
            for filename in os.listdir(path+target+"/ctl"):
                if filename.endswith('.fastq.gz') == False: continue
                jsonDict[f"chip.ctl_fastqs_rep{rep}_R1"] = [f"{path}{target}/ctl/{filename}"]
                rep += 1

    with open(f'{dest}{target}.json', 'w') as f:
            json.dump(jsonDict, f)