# meme 100_around_summits.fa -dna -revcomp -o meme_out

import os
import json
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from deeplift.visualization import viz_sequence
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

def fileToPSPM(filepath, title):
    lines = []
    for line in open(filepath):
        lines.append(line.rstrip().split('  '))
    pspm = np.array(lines).astype('float')
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(8,5), dpi=300)
    ax = fig.add_subplot(111)
    viz_sequence.plot_weights_given_ax(ax, pspm,
                                       height_padding_factor=0.2,
                                       length_padding=1.0,
                                       subticks_frequency=1.0,
                                       highlight={})
    fig.savefig(title+'.png', dpi=300)
    return pspm

PHO4_PBEXO_PSPM = fileToPSPM("../data/pho4_pbexo/meme_out/PSPM.txt", "PHO4_PBEXO_PSPM")
CBF1_PBEXO_PSPM = fileToPSPM("../data/cbf1_pbexo/meme_out/PSPM.txt", "CBF1_PBEXO_PSPM")
PHO4_NEXUS_PSPM = fileToPSPM("../data/nexus/120min/meme_out/PSPM.txt", "PHO4_NEXUS_PSPM")
CBF1_NEXUS_PSPM = fileToPSPM("../data/cbf1_nexus/meme_out/PSPM.txt", "CBF1_NEXUS_PSPM")

def probToPWM(matrix, title):
    pwm = np.log((matrix/0.25)+1e-6)
    centered_pwm = np.array([i-np.mean(i) for i in pwm])
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(8,5), dpi=300)
    ax = fig.add_subplot(111)
    viz_sequence.plot_weights_given_ax(ax, centered_pwm,
                                        height_padding_factor=0.2,
                                        length_padding=1.0,
                                        subticks_frequency=1.0,
                                        highlight={})
    fig.savefig(title+'.png', dpi=300)
    return pwm

keyToPWM = {}
keyToPWM["pho4_pbexo"] = probToPWM(PHO4_PBEXO_PSPM, "PHO4_PBEXO_PWM")
keyToPWM["cbf1_pbexo"] = probToPWM(CBF1_PBEXO_PSPM, "CBF1_PBEXO_PWM")
keyToPWM["pho4_nexus"] = probToPWM(PHO4_NEXUS_PSPM, "PHO4_NEXUS_PWM")
keyToPWM["cbf1_nexus"] = probToPWM(CBF1_NEXUS_PSPM, "CBF1_NEXUS_PWM")

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

for key in keyToPWM:
    yvals = []
    for flank in allFlanks:
        yvals.append(get_PWM_score(flank[:5]+"CACGTG"+flank[5:], keyToPWM[key]))
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
    meta["y-axis"] = "MEME based PWM scores -- scale = "+str(10.0**oom)
    meta["Number of points"] = len(xvals)
    meta["spearman"] = spearmanr(xvals, yvals)[0]
    meta["pearson"] = pearsonr(xvals, yvals)[0]
    # Residuals is sum of squared residuals of the least-squares fit
    meta["residuals"] = np.polyfit(xvals, yvals, 1, full=True)[1][0]
    # ax.set_xlim((1,5.75))
    # ax.set_ylim((1,5.75))
    # ax.set_aspect('equal')
    fig.savefig(key+'_MEME_baseline.png', dpi=300)
    plt.clf()
    with open(key+'_MEME_metadata.json', 'w') as fp: json.dump(meta, fp)