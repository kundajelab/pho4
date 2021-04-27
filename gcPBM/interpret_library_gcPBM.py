from __future__ import division, print_function, absolute_import
from collections import namedtuple
import keras_genomics
from keras_genomics.layers.convolutional import RevCompConv1D
from deeplift.dinuc_shuffle import dinuc_shuffle
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
import h5py
import codecs
import optparse
import os
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import pandas as pd
from math import log
import shap

parser = optparse.OptionParser()
parser.add_option('--target',
    action="store", dest="target",
    help="target", default=None)
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

target = options.target
key = target.split('_')[0]
dfs = pd.read_excel("/users/amr1/pho4/data/experimental/gcPBM/"+library[key])
all_xvals = dfs[column[key]]
seqs = dfs['Sequence']
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
    model = load_model('/users/amr1/pho4/data/models/'+target+'_model.h5')
    
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
with gzip.open('/users/amr1/pho4/data/gcpbm/'+target+'/idr.optimal_peak.narrowPeak.gz', 'rt') as inp:
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

def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in [0]:
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==2
        #At each position in the input sequence, we iterate over the one-hot encoding
        # possibilities (eg: for genomic sequence, this is ACGT i.e.
        # 1000, 0100, 0010 and 0001) and compute the hypothetical 
        # difference-from-reference in each case. We then multiply the hypothetical
        # differences-from-reference with the multipliers to get the hypothetical contributions.
        #For each of the one-hot encoding possibilities,
        # the hypothetical contributions are then summed across the ACGT axis to estimate
        # the total hypothetical contribution of each position. This per-position hypothetical
        # contribution is then assigned ("projected") onto whichever base was present in the
        # hypothetical sequence.
        #The reason this is a fast estimate of what the importance scores *would* look
        # like if different bases were present in the underlying sequence is that
        # the multipliers are computed once using the original sequence, and are not
        # computed again for each hypothetical sequence.
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:,i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference*mult[l]
            projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1) 
        to_return.append(np.mean(projected_hypothetical_contribs,axis=0))
    to_return.append(np.zeros_like(orig_inp[1]))
    return to_return

def shuffle_several_times(s):
    numshuffles=20
    return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)]),
            np.array([s[1] for i in range(numshuffles)])]

profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
    ([model.input[0], model.input[1]],
     tf.reduce_sum(model.outputs[0],axis=-1)),
    shuffle_several_times,
    combine_mult_and_diffref=combine_mult_and_diffref)

#See Google slide deck for explanations
#We meannorm as per section titled "Adjustments for Softmax Layers"
# in the DeepLIFT paper
meannormed_logits = (
    model.outputs[1]-
    tf.reduce_mean(model.outputs[1],axis=1)[:,None,:])
#'stop_gradient' will prevent importance from being propagated through
# this operation; we do this because we just want to treat the post-softmax
# probabilities as 'weights' on the different logits, without having the
# network explain how the probabilities themselves were derived
#Could be worth contrasting explanations derived with and without stop_gradient
# enabled...
stopgrad_meannormed_logits = tf.stop_gradient(meannormed_logits)
softmax_out = tf.nn.softmax(stopgrad_meannormed_logits,axis=1)
#Weight the logits according to the softmax probabilities, take the sum for each
# example. This mirrors what was done for the bpnet paper.
weightedsum_meannormed_logits = tf.reduce_sum(softmax_out*meannormed_logits,
                                              axis=(1,2))
profile_model_profile_explainer = shap.explainers.deep.TFDeepExplainer(
    ([model.input[0], model.input[2]],
     weightedsum_meannormed_logits),
    shuffle_several_times,
    combine_mult_and_diffref=combine_mult_and_diffref)

num_samples = 100
final_vals = []
preds_pre_seqs = []
preds_post_seqs = []
preds_logcount = []
preds_profile = []
counts_hypimps = []
profile_hypimps = []
counts_actualimps = []
profile_actualimps = []
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
    pre_seqs = getOneHot(pre_seqs)
    post_seqs = getOneHot(post_seqs)
    pre = model.predict([pre_seqs, np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    post = model.predict([post_seqs, np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    final_vals.append(np.mean(post[0]-pre[0]))
    preds_pre_seqs.append(pre_seqs)
    preds_post_seqs.append(post_seqs)
    preds_logcount.append(post[0])
    preds_profile.append(post[1])
    #The shap scores
    curr_post_counts_hypimps,_ = profile_model_counts_explainer.shap_values(
        [post_seqs, np.zeros((num_samples, 1))],
        progress_message=50)
    curr_post_profile_hypimps,_ = profile_model_profile_explainer.shap_values(
        [post_seqs, np.zeros((num_samples, out_pred_len, 2))],
        progress_message=50)
    curr_post_counts_hypimps = np.array(curr_post_counts_hypimps)
    curr_post_profile_hypimps = np.array(curr_post_profile_hypimps)
    counts_hypimps.append(curr_post_counts_hypimps)
    profile_hypimps.append(curr_post_profile_hypimps)
    counts_actualimps.append(curr_post_counts_hypimps*post_seqs)
    profile_actualimps.append(curr_post_profile_hypimps*post_seqs)

if not os.path.exists('preds-imp-scores/'+target):
    os.makedirs('preds-imp-scores/'+target)
np.save('preds-imp-scores/'+target+'/final_vals.npy', final_vals)
np.save('preds-imp-scores/'+target+'/preds_pre_seqs.npy', preds_pre_seqs)
np.save('preds-imp-scores/'+target+'/preds_post_seqs.npy', preds_post_seqs)
np.save('preds-imp-scores/'+target+'/preds_logcount.npy', preds_logcount)
np.save('preds-imp-scores/'+target+'/preds_profile.npy', preds_profile)
np.save('preds-imp-scores/'+target+'/counts_hypimps.npy', counts_hypimps)
np.save('preds-imp-scores/'+target+'/profile_hypimps.npy', profile_hypimps)
np.save('preds-imp-scores/'+target+'/counts_actualimps.npy', counts_actualimps)
np.save('preds-imp-scores/'+target+'/profile_actualimps.npy', profile_actualimps)