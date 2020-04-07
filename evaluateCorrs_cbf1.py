import os
import json
import codecs
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr

flankToCbf1Ddg = {}
firstLine = True
with open("data/experimental/all_predicted_ddGs.csv") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flank, Cbf1_ddg, Pho4_ddg = line.strip().split(',')
        flankToCbf1Ddg[flank] = float(Cbf1_ddg)

xvals_cbf1 = []
for key in flankToCbf1Ddg.keys():
    xvals_cbf1.append(flankToCbf1Ddg[key])

for filename in os.listdir("data/preds/"):
    if "cbf1" in filename and "fixed" not in filename:
        print(filename)
        obj_text = codecs.open("data/preds/"+filename, 'r', encoding='utf-8').read()
        flankToDeltaLogCount = json.loads(obj_text)
        yvals = []
        for key in flankToCbf1Ddg.keys():
            y_0 = np.array(flankToDeltaLogCount[key][0]).astype(float)
            y_1 = np.array(flankToDeltaLogCount[key][1]).astype(float)
            yvals.append(np.mean(y_1-y_0))
        plt.scatter(xvals_cbf1, yvals, alpha=0.1)
        plt.xlabel("DDG")
        plt.ylabel(filename)
        plt.title("DDG vs model predictions: "+str(spearmanr(xvals_cbf1, yvals)))
        plt.savefig('data/preds/figures/'+filename+'.png', bbox_inches='tight')
        plt.clf()
        plt.close()