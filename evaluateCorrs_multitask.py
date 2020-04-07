import os
import json
import codecs
import numpy as np
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
    if "multitask" in filename and "fixed" not in filename:
        print(filename)
        obj_text = codecs.open("data/preds/"+filename, 'r', encoding='utf-8').read()
        flankToDeltaLogCount = json.loads(obj_text)
        
        yvals_pho4 = []
        yvals_cbf1 = []
        for key in flankToPho4Ddg.keys():
            pho4_y_0 = np.array(flankToDeltaLogCount[key][0]).astype(float)
            cbf1_y_0 = np.array(flankToDeltaLogCount[key][1]).astype(float)
            pho4_y_1 = np.array(flankToDeltaLogCount[key][2]).astype(float)
            cbf1_y_1 = np.array(flankToDeltaLogCount[key][3]).astype(float)
            yvals_pho4.append(np.mean(pho4_y_1-pho4_y_0))
            yvals_cbf1.append(np.mean(cbf1_y_1-cbf1_y_0))
        
            plt.scatter(xvals_pho4, yvals_pho4, alpha=0.1)
            plt.xlabel("DDG")
            plt.ylabel(filename)
            plt.title("Pho4 DDG vs model predictions: "+str(spearmanr(xvals_pho4,yvals_pho4)))
            plt.savefig('data/preds/figures/pho4.'+filename+'.png', bbox_inches='tight')
            plt.clf()
            plt.close()

            plt.scatter(xvals_cbf1, yvals_cbf1, alpha=0.1)
            plt.xlabel("DDG")
            plt.ylabel(filename)
            plt.title("Cbf1 DDG vs model predictions: "+str(spearmanr(xvals_cbf1,yvals_cbf1)))
            plt.savefig('data/preds/figures/cbf1.'+filename+'.png', bbox_inches='tight')
            plt.clf()
            plt.close()