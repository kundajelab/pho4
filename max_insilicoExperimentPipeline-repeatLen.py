from collections import OrderedDict
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
from seqdataloader.batchproducers.coordbased.core import Coordinates
from seqdataloader.batchproducers.coordbased.coordstovals.core import CoordsToValsJoiner
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import AbstractCountAndProfileTransformer 
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import LogCountsPlusOne
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import SmoothProfiles
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import BigWigReader 
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import smooth_profiles
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import rolling_window
from seqdataloader.batchproducers.coordbased.coordstovals.bigwig import MultiTrackCountsAndProfile
from keras_genomics.layers.convolutional import RevCompConv1D
from seqdataloader.batchproducers.coordbased.core import Coordinates, KerasBatchGenerator, apply_mask
from deeplift.visualization import viz_sequence
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import keras
import keras.layers as kl
from keras.models import load_model
from keras.utils import CustomObjectScope
import shap
from deeplift.dinuc_shuffle import dinuc_shuffle
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path
import numpy as np
import subprocess
import gzip
import optparse
import os

parser = optparse.OptionParser()

parser.add_option('--gpus',
    action="store", dest="gpus",
    help="which gpus to use", default=None)
parser.add_option('--repeat',
    action="store", dest="repeat",
    help="what pattern to test", default=None)

options, args = parser.parse_args()

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

with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
    model = load_model("/users/amr1/pho4/data/models/max_hela_1_model.h5")

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
    
seq_len = 1346
out_pred_len = 1000
test_chrms = ["chr1", "chr8", "chr21"]
seq_peaks = []
with gzip.open("/users/amr1/pho4/data/gcpbm/max_hela_1/idr.optimal_peak.narrowPeak.gz", 'rt') as inp:
    for line in inp:
        chrm = line.strip().split('\t')[0]
        if chrm not in test_chrms:
            continue
        pStart = int(line.strip().split('\t')[1])
        summit = pStart + int(line.strip().split('\t')[-1])
        start = int(summit - (seq_len/2))
        end = int(summit + (seq_len/2))
        seq_peaks.append(GenomeDict[chrm][start:end].upper())

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
    flank = int((len(seq)-len(insert))/2.0)
    new_seq = seq[:flank]+insert+seq[flank+len(insert):]
    return new_seq

motif = "GTCACGTGAC"
candidates = []
for seq in seq_peaks:
    candidates.append(fill_into_center(dinuc_shuffle(seq), motif))
preds = model.predict([getOneHot(candidates), np.zeros((len(candidates),)), np.zeros((len(candidates),out_pred_len,2))])
count_preds = np.mean(preds[0], axis=1)
background = candidates[np.argmin(count_preds)]

mirror_dict = {'A':'T', 'C':'G', 'G':'C', 'T':'A'}

def getMirror(pattern):
    ret = ""
    for letter in pattern:
        ret += mirror_dict[letter]
    return ret[::-1]

def getRepeat(repeatLen, repeatPattern, stack='left', mirror=False):
    if stack == 'left':
        ret = repeatPattern * int(repeatLen/len(repeatPattern))
        if repeatLen%len(repeatPattern) != 0:
            ret = repeatPattern[-repeatLen%len(repeatPattern):] + ret
        return ret
    elif stack == 'right':
        ret = repeatPattern * int(repeatLen/len(repeatPattern))
        ret += repeatPattern[:repeatLen%len(repeatPattern)]
        return ret
    elif stack == 'both':
        if mirror:
            return (getRepeat(repeatLen, repeatPattern, stack='left', mirror=False),
                    getRepeat(repeatLen, getMirror(repeatPattern), stack='right', mirror=False))
        else:
            return (getRepeat(repeatLen, repeatPattern, stack='left', mirror=False),
                    getRepeat(repeatLen, repeatPattern, stack='right', mirror=False))
    else:
        print("unrecognized argument for stack")
        return None

bigFont = {'weight' : 'bold', 'size'   : 22}
smallFont = {'size' : 10}

repeatUntil = 65
repeatPattern = options.repeat
parentDir = "max_hela_1_preds_trends/"

def predictPlotAndSave(seqs, name):
    test_seqs = getOneHot(seqs)
    test_preds = model.predict([test_seqs, np.zeros((len(seqs),)),np.zeros((len(seqs),out_pred_len,2))])
    test_preds_logcount = np.mean(test_preds[0], axis = -1)
    np.save(parentDir+repeatPattern+'_'+name+'_preds.npy', test_preds_logcount) 

    fig = plt.figure(figsize=(10,9))
    plt.rc('font', **bigFont)
    plt.plot(range(len(seqs)), test_preds_logcount)
    plt.xlabel("repeat length")
    plt.ylabel("count prediction")
    plt.title(repeatPattern+" repeats "+name+" of motif")
    fig.savefig(parentDir+repeatPattern+'_'+name+'_trend.png')
    plt.close(fig)

# left only
seqs = []
for repeatLen in range(repeatUntil):
    repeat = getRepeat(repeatLen, repeatPattern, stack='left', mirror=False)
    insert = repeat + motif
    start = int((seq_len/2)-(len(motif)/2))-repeatLen
    end = int((seq_len/2)+(len(motif)/2))
    seqs.append(background[:start] + insert + background[end:])    
predictPlotAndSave(seqs, "left")

# right only
seqs = []
for repeatLen in range(repeatUntil):
    repeat = getRepeat(repeatLen, repeatPattern, stack='right', mirror=False)
    insert = motif + repeat
    start = int((seq_len/2)-(len(motif)/2))
    end = int((seq_len/2)+(len(motif)/2))+repeatLen
    seqs.append(background[:start] + insert + background[end:])    
predictPlotAndSave(seqs, "right")

# both
seqs = []
for repeatLen in range(repeatUntil):
    leftRepeat, rightRepeat = getRepeat(repeatLen, repeatPattern, stack='both', mirror=False)
    insert = leftRepeat + motif + rightRepeat
    start = int((seq_len/2)-(len(motif)/2))-repeatLen
    end = int((seq_len/2)+(len(motif)/2))+repeatLen
    seqs.append(background[:start] + insert + background[end:])    
predictPlotAndSave(seqs, "both")

# mirror both
seqs = []
for repeatLen in range(repeatUntil):
    leftRepeat, rightRepeat = getRepeat(repeatLen, repeatPattern, stack='both', mirror=True)
    insert = leftRepeat + motif + rightRepeat
    start = int((seq_len/2)-(len(motif)/2))-repeatLen
    end = int((seq_len/2)+(len(motif)/2))+repeatLen
    seqs.append(background[:start] + insert + background[end:])    
predictPlotAndSave(seqs, "both_mirror")