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
import math
import json
import gzip
import codecs
import os
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde

os.environ["CUDA_VISIBLE_DEVICES"]="3,5"

fastapath = "../data/genome/hg19/male.hg19.fa"
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

seq_len = 2114
out_pred_len = 1000
test_chrms = ["chr1", "chr8", "chr21"]
peaks = []
with gzip.open("../data/lncap_gr/idr.optimal_peak.narrowPeak.gz", 'rt') as inp:
    for line in inp:
        chrm = line.strip().split('\t')[0]
        if chrm not in test_chrms:
            continue
        pStart = int(line.strip().split('\t')[1])
        summit = pStart + int(line.strip().split('\t')[-1])
        start = int(summit - (seq_len/2))
        end = int(summit + (seq_len/2))
        new_seq = GenomeDict[chrm][start:end].upper()
        if len(new_seq) == seq_len: peaks.append(new_seq)

from deeplift.dinuc_shuffle import dinuc_shuffle

def fill_into_center(seq, insert):
    start = int((len(seq)/2.0)-35)
    new_seq = seq[:start]+"GTTCAGAGTTCTACAGTCCGACGATC"+  \
              seq[start+26:start+30]+insert+seq[start+45:start+49]+  \
              "TGGAATTCTCGGGTGCCAAGG"+seq[start+70:]
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

with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
    model = load_model("../data/models/lncap_gr_model.h5")

def sampleFromBins(allAffs, allSeqs):
    bins = np.linspace(np.min(allAffs), 1.001, 11)
    digitized = np.digitize(allAffs, bins)
    num_seqs = 1000
    sampled_keys = []
    curr_bin = 10
    curr_size = math.ceil(num_seqs/curr_bin)
    while num_seqs > 0 and curr_bin > 0:
        print(num_seqs, curr_bin, curr_size)
        large_basket = allSeqs[digitized == curr_bin]
        sample_size = min(curr_size, len(large_basket))
        sampled_keys.append(np.random.choice(large_basket, sample_size, replace=False))
        num_seqs -= sample_size
        curr_bin -= 1
        if curr_bin != 0: curr_size = math.ceil(num_seqs/curr_bin)
    sampled_keys = np.concatenate(np.array(sampled_keys))
    return sampled_keys

num_samples = 100
for key in ['GR_R2', 'GR_R3', 'GR_R4', 'GR_R5', 'GR_R6', 'GR_R7', 'GR_R8']:
    print(key)
    for table in ['kmerTableDivide.csv','kmerTableTransit.csv','symmetricKmerTableDivide.csv','symmetricKmerTableTransit.csv']:
        seqToAff = {}
        allSeqs = []
        allAffs = []
        firstLine = True
        with open("results/"+key+"/"+table) as inp:
            for line in inp:
                if firstLine:
                    firstLine = False
                    continue
                row = line.strip().split(',')
                if 'symmetric' in table and int(row[2]) == 0: continue
                seq = row[1][1:-1].upper()
                aff = float(row[-2])
                seqToAff[seq] = aff
                allSeqs.append(seq)
                allAffs.append(aff)

        sampled_keys = sampleFromBins(np.array(allAffs), np.array(allSeqs))
        xvals = []
        yvals = []
        seqToDeltaLogCount = {}
        for curr_seq in sampled_keys:
            pre_seqs = []
            post_seqs = []
            indices = np.random.choice(len(peaks), num_samples, replace=False)
            for idx in indices:
                pre_seq = dinuc_shuffle(peaks[idx])
                post_seq = fill_into_center(pre_seq, curr_seq)
                pre_seqs.append(pre_seq)
                post_seqs.append(post_seq)
            pre = model.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
            post = model.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
            seqToDeltaLogCount[curr_seq] = [pre[0].tolist(), post[0].tolist()]
            xvals.append(seqToAff[curr_seq])
            yvals.append(np.mean(post[0]-pre[0]))

        json.dump(seqToDeltaLogCount,
              codecs.open("preds/"+key+table+".json", 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True, indent=4)

        xy = np.vstack([xvals,yvals])
        z = gaussian_kde(xy)(xy)
        smallFont = {'size' : 10}
        plt.rc('font', **smallFont)
        fig, ax = plt.subplots()
        ax.scatter(xvals, yvals, c=z, edgecolor='', alpha=0.1)
        plt.xlabel("Affinities")
        plt.ylabel("Delta Log Counts")
        plt.title(key+table+" Affinities vs model predictions: "+str(spearmanr(xvals, yvals)))
        plt.savefig('figures/'+key+table+'_corrs.png', bbox_inches='tight')
        plt.clf()
        plt.close()
