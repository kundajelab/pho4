import os
import json
import math
import numpy as np
import modisco
import modisco.tfmodisco_workflow.workflow
from modisco.tfmodisco_workflow import workflow
import h5py
import modisco.util
from collections import Counter
from modisco.visualization import viz_sequence
import modisco.affinitymat.core
import modisco.cluster.phenograph.core
import modisco.cluster.phenograph.cluster
import modisco.cluster.core
import modisco.aggregator
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde
font = {'weight' : 'bold', 'size'   : 14}

flankToCbf1Ddg = {}
flankToPho4Ddg = {}
firstLine = True
allFlanks = []
with open("../data/experimental/all_scaled_nn_preds.txt") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flank,protein,_,_,_,ddG,_ = line.strip().split('\t')
        if protein == "Cbf1": flankToCbf1Ddg[flank] = float(ddG)
        else: flankToPho4Ddg[flank] = float(ddG)
        allFlanks.append(flank)
allFlanks = list(set(allFlanks))

post_hypimps = {}
post_actualimps = {}
seqs = {}
grp = {}
loaded_tfmodisco_results = {}

post_hypimps["pho4_pbexo"] = np.load("/users/amr1/pho4/data/imp-scores/pho4_pbexo/post_counts_hypimps.npy")[0]
post_actualimps["pho4_pbexo"] = np.load("/users/amr1/pho4/data/imp-scores/pho4_pbexo/post_counts_actualimps.npy")[0]
seqs["pho4_pbexo"] = np.load("/users/amr1/pho4/data/imp-scores/pho4_pbexo/seqs.npy")
grp["pho4_pbexo"] = h5py.File("/users/amr1/pho4/data/modisco/pho4_pbexo/pho4_counts/results.hdf5","r")

post_hypimps["cbf1_pbexo"] = np.load("/users/amr1/pho4/data/imp-scores/cbf1_pbexo/post_counts_hypimps.npy")
post_actualimps["cbf1_pbexo"] = np.load("/users/amr1/pho4/data/imp-scores/cbf1_pbexo/post_counts_actualimps.npy")  
seqs["cbf1_pbexo"] = np.load("/users/amr1/pho4/data/imp-scores/cbf1_pbexo/seqs.npy")
grp["cbf1_pbexo"] = h5py.File("/users/amr1/pho4/data/modisco/cbf1_pbexo/task0_counts_results.hdf5","r")

post_hypimps["pho4_nexus"] = np.load("/users/amr1/pho4/data/imp-scores/120min_pho4_nexus/post_counts_hypimps.npy")
post_actualimps["pho4_nexus"] = np.load("/users/amr1/pho4/data/imp-scores/120min_pho4_nexus/post_counts_actualimps.npy")  
seqs["pho4_nexus"] = np.load("/users/amr1/pho4/data/imp-scores/120min_pho4_nexus/seqs.npy")
grp["pho4_nexus"] = h5py.File("/users/amr1/pho4/data/modisco/120min_pho4_nexus/counts_results.hdf5","r")

post_hypimps["cbf1_nexus"] = np.load("/users/amr1/pho4/data/imp-scores/cbf1_nexus/post_counts_hypimps.npy")
post_actualimps["cbf1_nexus"] = np.load("/users/amr1/pho4/data/imp-scores/cbf1_nexus/post_counts_actualimps.npy")  
seqs["cbf1_nexus"] = np.load("/users/amr1/pho4/data/imp-scores/cbf1_nexus/seqs.npy")
grp["cbf1_nexus"] = h5py.File("/users/amr1/pho4/data/modisco/cbf1_nexus/counts_results.hdf5","r")

for key in seqs:
    print(key, post_actualimps[key].shape, post_hypimps[key].shape, seqs[key].shape)
    track_set = modisco.tfmodisco_workflow.workflow.prep_track_set(
                task_names=["task0"], contrib_scores={"task0": post_actualimps[key]},
                hypothetical_contribs={"task0": post_hypimps[key]}, one_hot=seqs[key])
    loaded_tfmodisco_results[key] = workflow.TfModiscoResults.from_hdf5(grp[key], track_set=track_set)
    grp[key].close()

from deeplift.visualization import viz_sequence

modisco_idx = {}
modisco_idx["pho4_pbexo"] = ("metacluster_1", 0, 12, 28)
modisco_idx["cbf1_pbexo"] = ("metacluster_1", 0, 17, 33)
modisco_idx["pho4_nexus"] = ("metacluster_0", 3, 12, 28)
modisco_idx["cbf1_nexus"] = ("metacluster_1", 0, 16, 32)

keyToCWM = {}
for key in seqs:
    untrimmed_pattern = (
        loaded_tfmodisco_results[key]
        .metacluster_idx_to_submetacluster_results[modisco_idx[key][0]]
        .seqlets_to_patterns_result.patterns[modisco_idx[key][1]])
    keyToCWM[key] = untrimmed_pattern["task0_hypothetical_contribs"].fwd[modisco_idx[key][2]:modisco_idx[key][3]]
    
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(8,5), dpi=300)
    ax = fig.add_subplot(111)
    viz_sequence.plot_weights_given_ax(ax, keyToCWM[key],
                                        height_padding_factor=0.2,
                                        length_padding=1.0,
                                        subticks_frequency=1.0,
                                        highlight={})
    fig.savefig(key+'_CWM.png', dpi=300)

def generate_matrix(seq):
    seq_matrix = np.zeros((4, len(seq)))
    for j in range(len(seq)):
        if seq[j] == 'A':
            seq_matrix[0,j] = 1
        elif seq[j] == 'C':
            seq_matrix[1,j] = 1
        elif seq[j] == 'G':
            seq_matrix[2,j] = 1
        elif seq[j] == 'T':
            seq_matrix[3,j] = 1
    return seq_matrix

def get_PWM_score(sequence, score_matrix):
    score = 0
    score_len = score_matrix.shape[0]
    for j in range(len(sequence) - score_len + 1):
        seq_matrix = generate_matrix(sequence[j:j+score_len])
        diagonal = np.diagonal(np.matmul(score_matrix, seq_matrix))
        score += np.prod(diagonal)
    return score

xvals_pho4 = []
xvals_cbf1 = []
for flank in allFlanks:
    xvals_pho4.append(flankToPho4Ddg[flank])
    xvals_cbf1.append(flankToCbf1Ddg[flank])

def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))

for key in keyToCWM:
    yvals = []
    for flank in allFlanks:
        yvals.append(get_PWM_score(flank[:5]+"CACGTG"+flank[5:], keyToCWM[key]))
    yvals = np.array(yvals)
    oom = orderOfMagnitude(np.max(np.abs(yvals)))
    yvals = yvals/(10.0**oom)
    if "pho4" in key: xvals = xvals_pho4
    else: xvals = xvals_cbf1
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111)
    plt.scatter(xvals, yvals, alpha=0.1)
    meta = {}
    meta["key"] = key
    meta["x-axis"] = "ddG"
    meta["y-axis"] = "MoDISco CWM scores -- scale = "+str(10.0**oom)
    meta["Number of points"] = len(xvals)
    meta["spearman"] = spearmanr(xvals, yvals)[0]
    meta["pearson"] = pearsonr(xvals, yvals)[0]
    # Residuals is sum of squared residuals of the least-squares fit
    meta["residuals"] = np.polyfit(xvals, yvals, 1, full=True)[1][0]
    # ax.set_xlim((1,5.75))
    # ax.set_ylim((1,5.75))
    # ax.set_aspect('equal')
    fig.savefig(key+'_MoDISco_baseline.png', dpi=300)
    plt.clf()
    with open(key+'_MoDISco_metadata.json', 'w') as fp: json.dump(meta, fp)