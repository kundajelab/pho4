import os
import re
import sys
import gzip
import json
import codecs
import pyBigWig
import subprocess
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy.stats import spearmanr
import optparse

parser = optparse.OptionParser()
parser.add_option('--targets',
    action="store", dest="targets",
    help="targets", default=None)
options, args = parser.parse_args()

targets = options.targets.split(',')

library = {
    "ets1": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log.xlsx",
    "elk1": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log.xlsx",
    "gabpa": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log.xlsx",
    "e2f1": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx",
    "e2f3": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx",
    "e2f4": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx",
    "max": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log.xlsx",
    "mxi": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log.xlsx",
    "myc": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log.xlsx",
    "runx1": "GSE97691_Combined_Runx1_10nM_50nM_Runx2_10nM_50nM_log.xlsx",
    "runx2": "GSE97691_Combined_Runx1_10nM_50nM_Runx2_10nM_50nM_log.xlsx"
}

column = {
    "ets1": "Ets1_100nM",
    "elk1": "Elk1_50nM",
    "gabpa": "Gabpa_100nM",
    "e2f1": "E2f1_250nM",
    "e2f3": "E2f3_250nM",
    "e2f4": "E2f4_500nM",
    "max": "Max",
    "mxi": "Mad_r",
    "myc": "Myc",
    "runx1": "Runx1_50nM",
    "runx2": "Runx2_50nM"
}

fastapath = "/users/amr1/pho4/data/genome/hg38/hg38.genome.fa"
GenomeDict={}
sequence=''
inputdatafile = open(fastapath)
for line in inputdatafile:
    if line[0]=='>':
        if sequence != '':
            GenomeDict[chrm] = ''.join(sequence)
        chrm = line.strip().split('>')[1]
        sequence=[]
        Keep=False
        continue
    else:
        sequence.append(line.strip())
GenomeDict[chrm] = ''.join(sequence)

for target in targets:
    key = target.split('_')[0]
    dfs = pd.read_excel("/users/amr1/pho4/data/experimental/gcPBM/"+library[key])
    all_xvals = dfs[column[key]]
    probes = dfs['Sequence']
    seqToDdg = {}
    for idx,probe in enumerate(probes):
        seqToDdg[probe] = all_xvals[idx]
    tfToBigWigs = (pyBigWig.open("/users/amr1/pho4/data/gcpbm/"+target+"/basename_prefix.pooled.positive.bigwig"),
                   pyBigWig.open("/users/amr1/pho4/data/gcpbm/"+target+"/basename_prefix.pooled.negative.bigwig"))
    
    seqToCoord = {}
    with gzip.open('/users/amr1/pho4/data/gcpbm/'+target+'/idr.optimal_peak.narrowPeak.gz', 'rt') as inp:
        for line in inp:
            chrm = line.strip().split('\t')[0]
            start = int(line.strip().split('\t')[1])-200
            end = int(line.strip().split('\t')[2])+200
            sequence = GenomeDict[chrm][start:end].upper()
            for probe in probes:
                loc = sequence.find(probe)
                if loc != -1:
                    if probe not in seqToCoord:
                        seqToCoord[probe] = []
                    seqToCoord[probe].append((chrm, start+loc))

    print("Number of probes in peaks: ", len(seqToCoord.keys()), ", total number of probes: ", len(probes))
    
    xvals = []
    for motif in seqToCoord.keys():
        xvals.append(seqToDdg[motif])

    seq_len = 201
    posFootprint = {}
    negFootprint = {}
    for motif in seqToCoord.keys():
        currentPosCounts= []
        currentNegCounts = []
        for chrm, motif_start in seqToCoord[motif]: 
            if "_" in chrm: continue
            center = motif_start+18
            start = int(center-(seq_len/2))
            end = int(center+(seq_len/2))
            posvals = np.array(tfToBigWigs[0].values(chrm, start, end))
            where_are_NaNs = np.isnan(posvals)
            posvals[where_are_NaNs] = 0.0
            currentPosCounts.append(posvals)
            negvals = np.array(tfToBigWigs[1].values(chrm, start, end))
            where_are_NaNs = np.isnan(negvals)
            negvals[where_are_NaNs] = 0.0
            currentNegCounts.append(negvals)
        posFootprint[motif] = np.mean(np.array(currentPosCounts), axis = 0)
        negFootprint[motif] = np.mean(np.array(currentNegCounts), axis = 0)
        
    window_sizes = [18, 36, 48, 64, 128, 150, 200]
    best_size = -1
    best_val = -1
    for window in window_sizes:
        start = int((seq_len/2)-(window/2))
        end = int((seq_len/2)+(window/2))
        yvals = []
        for flank in seqToCoord.keys():
            yvals.append(np.sum(posFootprint[flank][start:end]+ \
                                negFootprint[flank][start:end]))
        val = spearmanr(xvals, yvals)[0]
        if val > best_val:
            best_val = val
            best_size = window
            
    print("target: ", target, ", spearmanr: ", best_val, ", best window: ", best_size)