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
    if "multitask" in filename:
        print(filename)
        with open("data/preds/"+filename) as f:
            flankToLogCount = json.load(f)
            yvals_pho4 = []
            yvals_cbf1 = []
            for key in flankToPho4Ddg.keys():
                yvals_pho4.append(float(flankToLogCount[key][0]))
                yvals_cbf1.append(float(flankToLogCount[key][1]))

            plt.scatter(xvals_pho4, yvals_pho4, alpha=0.1)
            plt.xlabel("DDG")
            plt.ylabel(filename)
            plt.title("Pho4 DDG vs model predictions: "+str(spearmanr(xvals_pho4,yvals_pho4)))
            plt.savefig('data/preds/pho4.'+filename+'.png', bbox_inches='tight')
            plt.show()

            plt.scatter(xvals_cbf1, yvals_cbf1, alpha=0.1)
            plt.xlabel("DDG")
            plt.ylabel(filename)
            plt.title("Cbf1 DDG vs model predictions: "+str(spearmanr(xvals_cbf1,yvals_cbf1)))
            plt.savefig('data/preds/cbf1.'+filename+'.png', bbox_inches='tight')
            plt.show()