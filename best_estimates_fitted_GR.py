import os
import math
import json
import codecs
import numpy as np
from numpy import log, exp
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as LR
from scipy.optimize import minimize
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
import gzip
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

def sigmoid(x):
    return 1 / (1 + exp(-x))

def logit(p):
    return log(p) - log(1 - p)

class CalibratorFactory(object):
    def __call__(self, valid_preacts, valid_labels):
        raise NotImplementedError()

class LinearRegression(CalibratorFactory):
    def __init__(self, verbose=True):
        self.verbose = verbose 

    def __call__(self, valid_preacts, valid_labels):
        lr = LR().fit(valid_preacts.reshape(-1, 1), valid_labels)
    
        def calibration_func(preact):
            return lr.predict(preact.reshape(-1, 1))

        return calibration_func
    
class CalibratorFactory(object):
    def __call__(self, valid_preacts, valid_labels):
        raise NotImplementedError()

class SigmoidFit(CalibratorFactory):
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, valid_preacts, valid_labels):
        def loss_func(x):
            new_preacts = (x[0]*sigmoid(-valid_labels+x[1]))+x[2]
            return mean_squared_error(new_preacts, valid_preacts)

        x0 = np.array([1.0, 0.0, 0.0])
        res = None
        for c in range(10):
            curr = minimize(loss_func, x0, method='BFGS', options={'gtol': 1e-9, 'disp': True, 'maxiter': 1000})
            if res == None or curr.fun < res.fun:
                res = curr
        print("multiplier: ",res.x[0],", in sigmoid bias: ",res.x[1],", out of sigmoid bias: ",res.x[2])

        def calibration_func(label):
            return (res.x[0]*sigmoid(-label+res.x[1]))+res.x[2]

        def inv_func(preact):
            return -logit((preact-res.x[2])/res.x[0])+res.x[1]

        return calibration_func, inv_func
    
seqToDdg = {}
firstLine = True
with open("data/experimental/GR_bindingcurves_WT_1_out.csv") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        Oligo,Kd_estimate,ddG,Motif,Sequence = line.strip().split(',')
        seq = Sequence.upper()[14:30]
        if seq not in seqToDdg:
            seqToDdg[seq] = []
        seqToDdg[seq].append(float(ddG))
firstLine = True
with open("data/experimental/GR_bindingcurves_WT_2_out.csv") as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        Oligo,Kd_estimate,ddG,Motif,Sequence = line.strip().split(',')
        seq = Sequence.upper()[14:30]
        seqToDdg[seq].append(float(ddG))
valid_keys = np.random.choice(list(seqToDdg.keys()), 100, replace=False)
valid_labels = []
for curr_seq in valid_keys:
    valid_labels.append(np.mean(seqToDdg[curr_seq]))
valid_labels = np.array(valid_labels)
test_keys = []
test_labels = []
for curr_seq in seqToDdg.keys():
    if curr_seq in valid_keys: continue
    test_keys.append(curr_seq)
    test_labels.append(np.mean(seqToDdg[curr_seq]))
test_labels = np.array(test_labels)
    
keyToFiles = {}
keyToFiles["lncapGR"] = ("data/models/lncap_gr_model.h5",
                         "data/lncap_gr/idr.optimal_peak.narrowPeak.gz")
keyToFiles["exo_noCtl"] = ("data/models/noCtlK5_model.h5",
                     "data/noCtlK5/k5.bed.gz")
keyToFiles["seq_bjr"] = ("data/models/a5_seq_bjr_model.h5",
                       "data/a5/seq_bjr/idr_peaks.bed.gz")

seq_len = {}
seq_len["lncapGR"] = 2114
seq_len["exo_noCtl"] = 1346
seq_len["seq_bjr"] = 1346

fastapath = "data/genome/hg19/male.hg19.fa"
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

out_pred_len = 1000
test_chrms = ["chr1", "chr8", "chr21"]
keyToPeaks = {}
for key in keyToFiles: 
    peaks = []
    val0, val1 = keyToFiles[key]
    with gzip.open(val1, 'rt') as inp:
        for line in inp:
            chrm = line.strip().split('\t')[0]
            if chrm not in test_chrms:
                continue
            pStart = int(line.strip().split('\t')[1])
            summit = pStart + int(line.strip().split('\t')[-1])
            start = int(summit - (seq_len[key]/2))
            end = int(summit + (seq_len[key]/2))
            cadidate_seq = GenomeDict[chrm][start:end].upper()
            if len(cadidate_seq) == seq_len[key]: peaks.append(cadidate_seq)
    keyToPeaks[key] = peaks
    print(len(peaks))
    
from deeplift.dinuc_shuffle import dinuc_shuffle

def fill_into_center(seq, insert):
    start = int((len(seq)/2.0)-22)
    new_seq = seq[:start]+"CGCAATTGCGAGTC"+insert+"TCGACCTTCCTCTCCGGCGGTATGAC"+seq[start+56:]
    return new_seq

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
    

def plotCorrs(xvals, yvals, xlabel, ylabel, title):
    if np.isnan(xvals).any() or np.isinf(xvals).any(): return
    xy = np.vstack([xvals,yvals])
    z = gaussian_kde(xy)(xy)
    smallFont = {'size' : 10}
    plt.rc('font', **smallFont)
    fig, ax = plt.subplots()
    ax.scatter(xvals, yvals, c=z, edgecolor='', alpha=0.5)
    axes = plt.gca()
    p, residuals, _, _, _ = np.polyfit(xvals, yvals, 1, full=True)
    m, b = p
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.plot(X_plot, m*X_plot + b, '-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("spearman: "+str(spearmanr(xvals, yvals))+
              ", pearson: "+str(pearsonr(xvals, yvals))+
              ", residuals: "+str(residuals))
    plt.savefig('data/preds/figures/GR/'+title+'.png', bbox_inches='tight')
    plt.clf()
    plt.close()

num_samples = 100
for key in keyToFiles:
    print(key)
    val0, val1 = keyToFiles[key]
    with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
        model = load_model(val0)
    print(key+": loaded model")

    valid_preacts = []
    valid_log_preacts = []
    for curr_seq in valid_keys:
        pre_seqs = []
        post_seqs = []
        indices = np.random.choice(len(keyToPeaks[key]), num_samples, replace=False)
        for idx in indices:
            pre_seq = dinuc_shuffle(keyToPeaks[key][idx])
            post_seq = fill_into_center(pre_seq, curr_seq)
            pre_seqs.append(pre_seq)
            post_seqs.append(post_seq)
        if "noCtl" in key:
            pre = model.predict([getOneHot(pre_seqs)])
            post = model.predict([getOneHot(post_seqs)])
        else:
            pre = model.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
            post = model.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
        valid_log_preacts.append(np.mean(post[0]-pre[0]))
        valid_preacts.append(np.mean(np.exp(post[0])-np.exp(pre[0])))
    valid_log_preacts = np.array(valid_log_preacts)
    valid_preacts = np.array(valid_preacts)
    print("done with validation points")

    preacts = []
    log_preacts = []
    for curr_seq in test_keys:
        pre_seqs = []
        post_seqs = []
        indices = np.random.choice(len(keyToPeaks[key]), num_samples, replace=False)
        for idx in indices:
            pre_seq = dinuc_shuffle(keyToPeaks[key][idx])
            post_seq = fill_into_center(pre_seq, curr_seq)
            pre_seqs.append(pre_seq)
            post_seqs.append(post_seq)
        if "noCtl" in key:
            pre = model.predict([getOneHot(pre_seqs)])
            post = model.predict([getOneHot(post_seqs)])
        else:
            pre = model.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
            post = model.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
        log_preacts.append(np.mean(post[0]-pre[0]))
        preacts.append(np.mean(np.exp(post[0])-np.exp(pre[0])))
    log_preacts = np.array(log_preacts)
    preacts = np.array(preacts)
    print("done with test points")

    print(key+" linear fit with counts")
    lr = LinearRegression()
    calibration_func = lr(valid_preacts, valid_labels)
    calibrated_labels = calibration_func(preacts)
    plotCorrs(calibrated_labels, test_labels, "predicted ddG", "true ddG", key+"_LR")
    print(key+" linear fit with log counts")
    lr = LinearRegression()
    calibration_func = lr(valid_log_preacts, valid_labels)
    calibrated_labels = calibration_func(log_preacts)
    plotCorrs(calibrated_labels, test_labels, "predicted ddG", "true ddG", key+"_LR_log")
    print(key+" logit fit with counts")
    sf = SigmoidFit()
    calibration_func, inv_func = sf(valid_preacts, valid_labels)
    calibrated_labels = inv_func(preacts)
    plotCorrs(calibrated_labels, test_labels, "predicted ddG", "true ddG", key+"_SF")
    print(key+" logit fit with log counts")
    sf = SigmoidFit()
    calibration_func, inv_func = sf(valid_log_preacts, valid_labels)
    calibrated_labels = inv_func(log_preacts)
    plotCorrs(calibrated_labels, test_labels, "predicted ddG", "true ddG", key+"_SF_log")

    K.clear_session()
    del model