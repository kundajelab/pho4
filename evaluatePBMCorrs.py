import os
import json
import codecs
import numpy as np
from math import log10
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import optparse

parser = optparse.OptionParser()

parser.add_option('--measurements',
    action="store", dest="measurements",
    help="which file contains the measurements", default=None)

parser.add_option('--preds',
    action="store", dest="preds",
    help="which file contains the predictions", default=None)

options, args = parser.parse_args()

flankToSig = {}
firstLine = True
with open(options.measurements) as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flankToSig[line.strip().split('\t')[1]] = log10(float(line.strip().split('\t')[2]))

xvals = []
for key in flankToSig.keys():
    xvals.append(flankToSig[key])

obj_text = codecs.open("data/preds/"+options.preds, 'r', encoding='utf-8').read()
flankToCountPreds = json.loads(obj_text)

yvals = []
if "fixed" in options.preds:
    for key in flankToSig.keys():
        yvals.append(float(flankToCountPreds[key]))
else:
    for key in flankToSig.keys():
        y_0 = np.array(flankToCountPreds[key][0]).astype(float)
        y_1 = np.array(flankToCountPreds[key][1]).astype(float)
        yvals.append(np.mean(y_1-y_0))

xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
smallFont = {'size' : 10}
plt.rc('font', **smallFont)
fig, ax = plt.subplots()
ax.scatter(xvals, yvals, c=z, edgecolor='', alpha=0.1)
plt.xlabel("Log PBM Signal")
plt.ylabel(options.preds)
plt.title("PBM vs model predictions: "+str(spearmanr(xvals, yvals)))
plt.savefig('data/preds/figures/'+options.preds+'.png', bbox_inches='tight')
plt.clf()
plt.close()