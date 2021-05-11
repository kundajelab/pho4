import os
import json
import codecs
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import optparse

parser = optparse.OptionParser()

parser.add_option('--sample',
    action="store", dest="sample",
    help="sample or use the whole thing", default=None)
parser.add_option('--filename',
    action="store", dest="filename",
    help="which json to plot", default=None)

options, args = parser.parse_args()

flankToCbf1Ddg = {}
flankToPho4Ddg = {}
firstLine = True
allFlanks = []
with open("data/experimental/all_predicted_ddGs.csv") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flank, Cbf1_ddg, Pho4_ddg = line.strip().split(',')
        flankToCbf1Ddg[flank] = float(Cbf1_ddg)
        flankToPho4Ddg[flank] = float(Pho4_ddg)
        allFlanks.append(flank)

if options.sample == "True" or options.sample == "1":
    extension = '.sample.png'
    sampled_keys = np.random.choice(allFlanks, 50000, replace=False)
else:
    extension = '.png'
    sampled_keys = allFlanks

xvals_pho4 = []
xvals_cbf1 = []
for key in sampled_keys:
    xvals_pho4.append(flankToPho4Ddg[key])
    xvals_cbf1.append(flankToCbf1Ddg[key])

obj_text = codecs.open("data/preds/"+options.filename, 'r', encoding='utf-8').read()
flankToCountPreds = json.loads(obj_text)

if "multitask" not in options.filename:
    if "pho4" in options.filename: xvals = xvals_pho4
    elif "cbf1" in options.filename: xvals = xvals_cbf1
    else: xvals = []
    yvals = []
    if "fixed" in options.filename:
        for key in sampled_keys:
            yvals.append(float(flankToCountPreds[key]))
    else:
        for key in sampled_keys:
            y_0 = np.array(flankToCountPreds[key][0]).astype(float)
            y_1 = np.array(flankToCountPreds[key][1]).astype(float)
            yvals.append(np.mean(y_1-y_0))
    print("DDG vs model predictions: "+str(spearmanr(xvals, yvals)))
    xy = np.vstack([xvals,yvals])
    z = gaussian_kde(xy)(xy)
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 18}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)
    ax.scatter(xvals, yvals, c=z, edgecolor='', alpha=0.1)
    fig.savefig('data/preds/figures/'+options.filename+extension, dpi=300)
    #plt.xlabel("DDG")
    #plt.ylabel(options.filename)
    #plt.title("DDG vs model predictions: "+str(spearmanr(xvals, yvals)))
    #plt.clf()
    #plt.close()
else:
    yvals_pho4 = []
    yvals_cbf1 = []
    if "fixed" in options.filename:
        for key in sampled_keys:
            yvals_pho4.append(float(flankToCountPreds[key][0]))
            yvals_cbf1.append(float(flankToCountPreds[key][1]))
    else:
        for key in sampled_keys:
            pho4_y_0 = np.array(flankToCountPreds[key][0]).astype(float)
            cbf1_y_0 = np.array(flankToCountPreds[key][1]).astype(float)
            pho4_y_1 = np.array(flankToCountPreds[key][2]).astype(float)
            cbf1_y_1 = np.array(flankToCountPreds[key][3]).astype(float)
            yvals_pho4.append(np.mean(pho4_y_1-pho4_y_0))
            yvals_cbf1.append(np.mean(cbf1_y_1-cbf1_y_0))

    xy = np.vstack([xvals_pho4,yvals_pho4])
    z = gaussian_kde(xy)(xy)
    smallFont = {'size' : 10}
    plt.rc('font', **smallFont)
    fig, ax = plt.subplots()
    ax.scatter(xvals_pho4, yvals_pho4, c=z, edgecolor='', alpha=0.1)
    plt.xlabel("DDG")
    plt.ylabel(options.filename)
    plt.title("DDG vs model predictions: "+str(spearmanr(xvals_pho4, yvals_pho4)))
    plt.savefig('data/preds/figures/'+options.filename+extension, bbox_inches='tight')
    plt.clf()
    plt.close()

    xy = np.vstack([xvals_cbf1,yvals_cbf1])
    z = gaussian_kde(xy)(xy)
    smallFont = {'size' : 10}
    plt.rc('font', **smallFont)
    fig, ax = plt.subplots()
    ax.scatter(xvals_cbf1, yvals_cbf1, c=z, edgecolor='', alpha=0.1)
    plt.xlabel("DDG")
    plt.ylabel(options.filename)
    plt.title("DDG vs model predictions: "+str(spearmanr(xvals_cbf1, yvals_cbf1)))
    plt.savefig('data/preds/figures/'+options.filename+extension, bbox_inches='tight')
    plt.clf()
    plt.close()