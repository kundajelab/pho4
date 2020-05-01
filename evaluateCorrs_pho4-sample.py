import os
import json
import codecs
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde

flankToPho4Ddg = {}
firstLine = True
allFlanks = []
with open("data/experimental/all_predicted_ddGs.csv") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flank, Cbf1_ddg, Pho4_ddg = line.strip().split(',')
        flankToPho4Ddg[flank] = float(Pho4_ddg)
        allFlanks.append(flank)

sampled_keys = np.random.choice(allFlanks, 50000, replace=False)
xvals_pho4 = []
for key in sampled_keys:
    xvals_pho4.append(flankToPho4Ddg[key])

for filename in os.listdir("data/preds/"):
    if "pho4" in filename and "fixed" not in filename:
        print(filename)
        obj_text = codecs.open("data/preds/"+filename, 'r', encoding='utf-8').read()
        flankToDeltaLogCount = json.loads(obj_text)
        yvals = []
        for key in sampled_keys:
            y_0 = np.array(flankToDeltaLogCount[key][0]).astype(float)
            y_1 = np.array(flankToDeltaLogCount[key][1]).astype(float)
            yvals.append(np.mean(y_1-y_0))
        xy = np.vstack([xvals_pho4,yvals])
        z = gaussian_kde(xy)(xy)
        smallFont = {'size' : 10}
        plt.rc('font', **smallFont)
        fig, ax = plt.subplots()
        ax.scatter(xvals_pho4, yvals, c=z, edgecolor='', alpha=0.1)
        plt.xlabel("DDG")
        plt.ylabel(filename)
        plt.title("DDG vs model predictions: "+str(spearmanr(xvals_pho4, yvals)))
        plt.savefig('data/preds/figures/'+filename+'_sample.png', bbox_inches='tight')
        plt.clf()
        plt.close()