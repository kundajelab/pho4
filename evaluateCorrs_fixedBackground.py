import os
import json
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr

flankToCbf1Ddg = {}
flankToPho4Ddg = {}
firstLine = True
with open("data/experimental/all_predicted_ddGs.csv") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flank, Cbf1_ddg, Pho4_ddg = line.strip().split(',')
        flankToCbf1Ddg[flank] = float(Cbf1_ddg)
        flankToPho4Ddg[flank] = float(Pho4_ddg)

xvals_pho4 = []
xvals_cbf1 = []
for key in flankToPho4Ddg.keys():
    xvals_pho4.append(flankToPho4Ddg[key])
    xvals_cbf1.append(flankToCbf1Ddg[key])
    
for filename in os.listdir("data/preds/"):
    if "cbf1" in filename:
        xvals = xvals_cbf1
    elif "pho4" in filename:
        xvals = xvals_pho4
    else:
        continue
    print(filename)
    with open("data/preds/"+filename) as f:
        flankToLogCount = json.load(f)
        yvals = []
        for key in flankToPho4Ddg.keys():
            yvals.append(flankToLogCount[key])
        plt.scatter(xvals, yvals, alpha=0.1)
        plt.xlabel("DDG")
        plt.ylabel(filename)
        plt.title("DDG vs model predictions: "+str(spearmanr(xvals,yvals)))
        plt.savefig('data/preds/'+filename+'.png', bbox_inches='tight')
        plt.show()