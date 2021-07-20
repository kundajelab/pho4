import os
import json
import codecs
import numpy as np
import matplotlib
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
with open("data/experimental/all_scaled_nn_preds.txt") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flank,protein,_,_,_,ddG,_ = line.strip().split('\t')
        if protein == "Cbf1": flankToCbf1Ddg[flank] = float(ddG)
        else: flankToPho4Ddg[flank] = float(ddG)
        allFlanks.append(flank)
allFlanks = list(set(allFlanks))

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

if "pho4" in options.filename: xvals = xvals_pho4
elif "cbf1" in options.filename: xvals = xvals_cbf1
else: xvals = []
yvals = []
for key in sampled_keys:
    y_0 = np.array(flankToCountPreds[key][0]).astype(float)
    y_1 = np.array(flankToCountPreds[key][1]).astype(float)
    yvals.append(np.mean(y_1-y_0))

xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
font = {'weight' : 'bold',
        'size'   : 18}
plt.rc('font', **font)
fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(111)
ax.scatter(xvals, yvals, c=z, edgecolor='', alpha=0.1)
fig.savefig('data/preds/figures/'+options.filename+extension, dpi=300)
meta = {}
meta["key"] = key
meta["x-axis"] = "ddG"
meta["y-axis"] = "marginalization scores"
meta["Number of points"] = len(xvals)
meta["spearman"] = spearmanr(xvals, yvals)[0]
meta["pearson"] = pearsonr(xvals, yvals)[0]
# Residuals is sum of squared residuals of the least-squares fit
meta["residuals"] = np.polyfit(xvals, yvals, 1, full=True)[1][0]
with open('data/preds/figures/'+options.filename+extension+'.metadata.json', 'w') as fp: json.dump(meta, fp)