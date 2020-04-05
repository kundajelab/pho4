import os
import json
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr

flankToPho4Ddg = {}
firstLine = True
with open("data/experimental/all_predicted_ddGs.csv") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flank, Cbf1_ddg, Pho4_ddg = line.strip().split(',')
        flankToPho4Ddg[flank] = float(Pho4_ddg)

xvals_pho4 = []
for key in flankToPho4Ddg.keys():
    xvals_pho4.append(flankToPho4Ddg[key])
    
for filename in os.listdir("data/preds/"):
    if "pho4" in filename:
        print(filename)
        with open("data/preds/"+filename) as f:
            flankToLogCount = json.load(f)
            yvals = []
            for key in flankToPho4Ddg.keys():
                yvals.append(flankToLogCount[key])
            plt.scatter(xvals_pho4, yvals, alpha=0.1)
            plt.xlabel("DDG")
            plt.ylabel(filename)
            plt.title("DDG vs model predictions: "+str(spearmanr(xvals_pho4,yvals)))
            plt.savefig('data/preds/figures/'+filename+'.png', bbox_inches='tight')
            plt.clf()
            plt.close()