import os
import json
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
    if "cbf1" in filename:
        print(filename)
        with open("data/preds/"+filename) as f:
            flankToLogCount = json.load(f)
            yvals = []
            for key in flankToCbf1Ddg.keys():
                yvals.append(flankToLogCount[key])
            plt.scatter(xvals_cbf1, yvals, alpha=0.1)
            plt.xlabel("DDG")
            plt.ylabel(filename)
            plt.title("DDG vs model predictions: "+str(spearmanr(xvals_cbf1,yvals)))
            plt.savefig('data/preds/figures/'+filename+'.png', bbox_inches='tight')
            plt.clf()
            plt.close()