import keras_genomics
from keras_genomics.layers.convolutional import RevCompConv1D
import keras
import keras.layers as kl
from keras import backend as K 
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
from keras.models import load_model
from keras.utils import CustomObjectScope
import json
import gzip
import codecs
import os
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

flankToCbf1Ddg = {}
flankToPho4Ddg = {}
firstLine = True
allFlanks = []
with open("../data/experimental/all_predicted_ddGs.csv") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flank, Cbf1_ddg, Pho4_ddg = line.strip().split(',')
        flankToCbf1Ddg[flank] = float(Cbf1_ddg)
        flankToPho4Ddg[flank] = float(Pho4_ddg)
        allFlanks.append(flank)

sampled_keys = np.random.choice(allFlanks, 25000, replace=False)
xvals_pho4 = []
xvals_cbf1 = []
for key in sampled_keys:
    xvals_pho4.append(flankToPho4Ddg[key])
    xvals_cbf1.append(flankToCbf1Ddg[key])

keyToFiles = {}
keyToFiles["pho4_pbexo_standard"] = ("../data/models/pho4_pbexo_model.h5",
                                     "../data/pho4_pbexo/pho4.pbexo.bed",
                                     546, 200)
keyToFiles["pho4_pbexo_matched"] = ("../data/models/pho4_pbexo_matched_model.h5",
                                     "../data/pho4_pbexo/pho4.pbexo.bed",
                                     546, 200)
keyToFiles["cbf1_pbexo_standard"] = ("../data/models/cbf1_pbexo_model.h5",
                                     "../data/cbf1_pbexo/cbf1.pbexo.bed",
                                     546, 200)
keyToFiles["cbf1_pbexo_matched"] = ("../data/models/cbf1_pbexo_matched_model.h5",
                                     "../data/cbf1_pbexo/cbf1.pbexo.bed",
                                     546, 200)
keyToFiles["cbf1_chipexo_standard"] = ("../data/models/cbf1_chipexo_model.h5",
                                       "../data/cbf1_chipexo/cbf1.chipexo.bed",
                                       546, 225)
keyToFiles["cbf1_chipexo_matched"] = ("../data/models/cbf1_chipexo_matched_model.h5",
                                       "../data/cbf1_chipexo/cbf1.chipexo.bed",
                                       546, 225)

fastapath = "../data/genome/saccer/sacCer3.genome.fa"
GenomeDict={}
sequence=''
inputdatafile = open(fastapath)
for line in inputdatafile:
    if line[0]=='>':
        if sequence != '':
            GenomeDict[chrm] = ''.join(sequence)
        chrm = line.strip().split('>')[1]
        sequence=[]
        Keep=False
        continue
    else:
        sequence.append(line.strip())
GenomeDict[chrm] = ''.join(sequence)

test_chrms = ["chrX", "chrXI"]
keyToPeaks = {}
for key in keyToFiles: 
    peaks = []
    val1 = keyToFiles[key][1]
    seq_len = keyToFiles[key][2]
    with open(val1) as inp:
        for line in inp:
            chrm = line.strip().split('\t')[0]
            if chrm not in test_chrms:
                continue
            pStart = int(line.strip().split('\t')[1])
            summit = pStart + 1
            start = int(summit - (seq_len/2))
            end = int(summit + (seq_len/2))
            current_seq = GenomeDict[chrm][start:end].upper()
            if len(current_seq) == seq_len: peaks.append(current_seq)
    keyToPeaks[key] = peaks

from deeplift.dinuc_shuffle import dinuc_shuffle

ltrdict = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],
           'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],
           'T':[0,0,0,1],'N':[0,0,0,0]}
def getOneHot(ISM_sequences):
  # takes in list of sequences
    one_hot_seqs = []
    for seq in ISM_sequences:
        one_hot = []
        for i in range(len(seq)):
            one_hot.append(ltrdict[seq[i:i+1]])
        one_hot_seqs.append(one_hot)
    return np.array(one_hot_seqs)

def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                         logits=logits)
    return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
            tf.to_float(tf.shape(true_counts)[0]))

#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
class MultichannelMultinomialNLL(object):
    def __init__(self, n):
        self.__name__ = "MultichannelMultinomialNLL"
        self.n = n

    def __call__(self, true_counts, logits):
        for i in range(self.n):
            loss = multinomial_nll(true_counts[..., i], logits[..., i])
            if i == 0:
                total = loss
            else:
                total += loss
        return total

    def get_config(self):
        return {"n": self.n}

num_samples = 20
for key in keyToFiles:
    print(key)
    val0 = keyToFiles[key][0]
    seq_len = keyToFiles[key][2]
    out_pred_len = keyToFiles[key][3]
    with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
        model = load_model(val0)
    yvals = []
    for flank in sampled_keys:
        pre_seqs = []
        post_seqs = []
        insert = flank[:5] + "CACGTG" + flank[5:]
        insert_len = len(insert)
        start = int((seq_len/2)-(insert_len/2))
        indices = np.random.choice(len(keyToPeaks[key]), num_samples, replace=False)
        for idx in indices:
            pre_seq = dinuc_shuffle(keyToPeaks[key][idx])
            post_seq = pre_seq[:start] + insert + pre_seq[start+insert_len:]
            pre_seqs.append(pre_seq)
            post_seqs.append(post_seq)
        pre = model.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
        post = model.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
        yvals.append(np.mean(post[0]-pre[0]))

    K.clear_session()
    del model

    if "pho4" in key: xvals = xvals_pho4
    else: xvals = xvals_cbf1
    xy = np.vstack([xvals,yvals])
    z = gaussian_kde(xy)(xy)
    smallFont = {'size' : 10}
    plt.rc('font', **smallFont)
    fig, ax = plt.subplots()
    ax.scatter(xvals, yvals, c=z, edgecolor='', alpha=0.5)
    plt.xlabel("DDG")
    plt.ylabel("Delta Log Counts")
    plt.title(key+" DDG vs model predictions: "+str(spearmanr(xvals, yvals)))
    plt.savefig("figs/"+key+".png")