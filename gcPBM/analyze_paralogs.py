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
from deeplift.dinuc_shuffle import dinuc_shuffle
import pandas as pd
import h5py
import json
import os
import gzip
from math import log
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import optparse

parser = optparse.OptionParser()

parser.add_option('--target1',
    action="store", dest="target1",
    help="target1", default=None)
parser.add_option('--target2',
    action="store", dest="target2",
    help="target2", default=None)
parser.add_option('--gpus',
    action="store", dest="gpus",
    help="gpus", default=None)
options, args = parser.parse_args()

library = {
    "ets1": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log.xlsx",
    "elk1": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log.xlsx",
    "gabpa": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log.xlsx",
    "e2f1": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx",
    "e2f3": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx",
    "e2f4": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx",
    "max": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log.xlsx",
    "mxi": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log.xlsx",
    "myc": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log.xlsx",
    "runx1": "GSE97691_Combined_Runx1_10nM_50nM_Runx2_10nM_50nM_log.xlsx",
    "runx2": "GSE97691_Combined_Runx1_10nM_50nM_Runx2_10nM_50nM_log.xlsx"
}
column = {
    "ets1": "Ets1_100nM",
    "elk1": "Elk1_50nM",
    "gabpa": "Gabpa_100nM",
    "e2f1": "E2f1_250nM",
    "e2f3": "E2f3_250nM",
    "e2f4": "E2f4_500nM",
    "max": "Max",
    "mxi": "Mad_r",
    "myc": "Myc",
    "runx1": "Runx1_50nM",
    "runx2": "Runx2_50nM"
}

target1 = options.target1
if target1[0].isdigit():
    key1 = target1.split('_')[1].lower()
    if key1[-1] == '-': key1 = key1[:-1]
else: key1 = target1.split('_')[0]
dfs = pd.read_excel("/users/amr1/pho4/data/experimental/gcPBM/"+library[key1])
all_xvals1 = dfs[column[key1]]

target2 = options.target2
if target2[0].isdigit():
    key2 = target2.split('_')[1].lower()
    if key2[-1] == '-': key2 = key2[:-1]
else: key2 = target2.split('_')[0]
dfs = pd.read_excel("/users/amr1/pho4/data/experimental/gcPBM/"+library[key2])
all_xvals2 = dfs[column[key2]]

seqs = dfs['Sequence']

fastapath = "/users/amr1/pho4/data/genome/hg38/hg38.genome.fa"
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

os.environ["CUDA_VISIBLE_DEVICES"]=options.gpus
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
    
ltrdict = {
           'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],
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

def fill_into_center(seq, insert):
    pos = int((len(seq)-len(insert))/2.0)
    new_seq = seq[:pos] + insert + seq[pos+len(insert):]
    return new_seq

from deeplift.dinuc_shuffle import dinuc_shuffle
    
with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
    model1 = load_model('/users/amr1/pho4/data/models/'+target1+'_model.h5')
        
seq_len = 1346
out_pred_len = 1000
test_chrms = ["chr1", "chr8", "chr21"]
seq_peaks = []
with gzip.open('/users/amr1/pho4/data/gcpbm/'+target1+'/idr.optimal_peak.narrowPeak.gz', 'rt') as inp:
    for line in inp:
        chrm = line.strip().split('\t')[0]
        if chrm not in test_chrms:
            continue
        pStart = int(line.strip().split('\t')[1])
        summit = pStart + int(line.strip().split('\t')[-1])
        start = int(summit - (seq_len/2))
        end = int(summit + (seq_len/2))
        candidate_seq = GenomeDict[chrm][start:end].upper()
        if len(candidate_seq) == seq_len: seq_peaks.append(candidate_seq)

num_samples = min(100, len(seq_peaks)) 
yvals1 = []
for idx, insert in enumerate(seqs):
    if idx % 1000 == 0:
        print("Done with ", idx)
    pre_seqs = []
    post_seqs = []
    indices = np.random.choice(len(seq_peaks), num_samples, replace=False)
    for idx in indices:
        pre_seq = dinuc_shuffle(seq_peaks[idx])
        post_seq = fill_into_center(pre_seq, insert)
        pre_seqs.append(pre_seq)
        post_seqs.append(post_seq)
    pre = model1.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    post = model1.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    yvals1.append(np.mean(post[0]-pre[0]))
    
with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
    model1 = load_model('/users/amr1/pho4/data/models/'+target1+'_model.h5')
        
seq_len = 1346
out_pred_len = 1000
test_chrms = ["chr1", "chr8", "chr21"]
seq_peaks = []
with gzip.open('/users/amr1/pho4/data/gcpbm/'+target1+'/idr.optimal_peak.narrowPeak.gz', 'rt') as inp:
    for line in inp:
        chrm = line.strip().split('\t')[0]
        if chrm not in test_chrms:
            continue
        pStart = int(line.strip().split('\t')[1])
        summit = pStart + int(line.strip().split('\t')[-1])
        start = int(summit - (seq_len/2))
        end = int(summit + (seq_len/2))
        candidate_seq = GenomeDict[chrm][start:end].upper()
        if len(candidate_seq) == seq_len: seq_peaks.append(candidate_seq)

num_samples = min(100, len(seq_peaks)) 
yvals1 = []
for idx, insert in enumerate(seqs):
    if idx % 1000 == 0:
        print("Done with ", idx)
    pre_seqs = []
    post_seqs = []
    indices = np.random.choice(len(seq_peaks), num_samples, replace=False)
    for idx in indices:
        pre_seq = dinuc_shuffle(seq_peaks[idx])
        post_seq = fill_into_center(pre_seq, insert)
        pre_seqs.append(pre_seq)
        post_seqs.append(post_seq)
    pre = model1.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    post = model1.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    yvals1.append(np.mean(post[0]-pre[0]))
    
K.clear_session()
del model1

with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
    model2 = load_model('/users/amr1/pho4/data/models/'+target2+'_model.h5')
        
seq_len = 1346
out_pred_len = 1000
test_chrms = ["chr1", "chr8", "chr21"]
seq_peaks = []
with gzip.open('/users/amr1/pho4/data/gcpbm/'+target2+'/idr.optimal_peak.narrowPeak.gz', 'rt') as inp:
    for line in inp:
        chrm = line.strip().split('\t')[0]
        if chrm not in test_chrms:
            continue
        pStart = int(line.strip().split('\t')[1])
        summit = pStart + int(line.strip().split('\t')[-1])
        start = int(summit - (seq_len/2))
        end = int(summit + (seq_len/2))
        candidate_seq = GenomeDict[chrm][start:end].upper()
        if len(candidate_seq) == seq_len: seq_peaks.append(candidate_seq)

num_samples = min(100, len(seq_peaks)) 
yvals2 = []
for idx, insert in enumerate(seqs):
    if idx % 1000 == 0:
        print("Done with ", idx)
    pre_seqs = []
    post_seqs = []
    indices = np.random.choice(len(seq_peaks), num_samples, replace=False)
    for idx in indices:
        pre_seq = dinuc_shuffle(seq_peaks[idx])
        post_seq = fill_into_center(pre_seq, insert)
        pre_seqs.append(pre_seq)
        post_seqs.append(post_seq)
    pre = model2.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    post = model2.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    yvals2.append(np.mean(post[0]-pre[0]))
    
K.clear_session()
del model2

xy = np.vstack([all_xvals1,yvals2])
z = gaussian_kde(xy)(xy)
smallFont = {'size' : 10}
plt.rc('font', **smallFont)
fig, ax = plt.subplots()
ax.scatter(all_xvals1, yvals2, c=z, edgecolor='', alpha=0.5)
plt.xlabel(key1 + " Log gcPBM Signal")
plt.ylabel(target2 + " Delta Log Counts")
plt.title("spearman: "+str(spearmanr(all_xvals1, yvals2)[0])+
          ", pearson: "+ str(pearsonr(all_xvals1, yvals2)[0]))
fig.savefig('preds/'+key1+'_v_'+target2+'_eval.png', dpi=fig.dpi)

xy = np.vstack([all_xvals2,yvals1])
z = gaussian_kde(xy)(xy)
smallFont = {'size' : 10}
plt.rc('font', **smallFont)
fig, ax = plt.subplots()
ax.scatter(all_xvals2, yvals1, c=z, edgecolor='', alpha=0.5)
plt.xlabel(key2 + " Log gcPBM Signal")
plt.ylabel(target1 + " Delta Log Counts")
plt.title("spearman: "+str(spearmanr(all_xvals2, yvals1)[0])+
          ", pearson: "+ str(pearsonr(all_xvals2, yvals1)[0]))
fig.savefig('preds/'+key2+'_v_'+target1+'_eval.png', dpi=fig.dpi)