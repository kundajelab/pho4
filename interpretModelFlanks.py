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
import keras
import keras.layers as kl
from keras.models import load_model
from keras.utils import CustomObjectScope
import shap
from deeplift.dinuc_shuffle import dinuc_shuffle
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import optparse
import os

parser = optparse.OptionParser()

parser.add_option('--gpus',
    action="store", dest="gpus",
    help="which gpus to use", default=None)
parser.add_option('--task_names',
    action="store", dest="task_names",
    help="what are the tasks", default=None)
parser.add_option('--out_pred_len',
    action="store", dest="out_pred_len",
    help="length of predicted profile", default=None)
parser.add_option('--flanks_csv',
    action="store", dest="flanks_csv",
    help="where are the flanks", default=None)
parser.add_option('--peaks_bed',
    action="store", dest="peaks_bed",
    help="where are the peaks", default=None)
parser.add_option('--model',
    action="store", dest="model",
    help="where is the model", default=None)
parser.add_option('--output_dir',
    action="store", dest="output_dir",
    help="where to store end result", default=None)

options, args = parser.parse_args()

flanks = []
firstLine = True
with open(options.flanks_csv) as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flanks.append(line.strip().split(',')[0])

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
    model = load_model(options.model)

seq_len = 546
out_pred_len = int(options.out_pred_len)
task_names = options.task_names.split(',')
test_chrms = ["chrX", "chrXI"]
peaks = []
with open(options.peaks_bed) as inp:
    for line in inp:
        chrm = line.strip().split('\t')[0]
        if chrm not in test_chrms:
            continue
        pStart = int(line.strip().split('\t')[1])
        summit = pStart + 1
        start = int(summit - (seq_len/2))
        end = int(summit + (seq_len/2))
        peaks.append((chrm, start, end))

def readChromSizes(chrom_sizes_file):
    chrom_size_list = []
    for line in open(chrom_sizes_file):
        (chrom, size) = line.rstrip().split("\t")[0:2]
        chrom_size_list.append((chrom,int(size)))
    return chrom_size_list

chrms = ["chrI","chrII","chrIII","chrIV","chrV","chrVI","chrVII","chrVIII",
         "chrIX","chrX","chrXI","chrXII","chrXIII","chrXIV","chrXV","chrXVI","chrM"]

def customChromSizeSort(c):
    return chrms.index(c[0])

from pyfaidx import Fasta
genome_object = Fasta("/users/amr1/pho4/data/genome/sacCer3.genome.fa")

chrom_sizes = readChromSizes("/users/amr1/pho4/data/genome/sacCer3.chrom.sizes")
chrom_sizes.sort(key=customChromSizeSort)

num_chroms = len(chrom_sizes)

fasta_sequences = []
for chrom in chrom_sizes:
    chrom_num = chrom[0]
    chrom_size = chrom[1]
    fasta_sequences.append(genome_object[chrom_num][0:chrom_size].seq)
    
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

seq_peaks = []
for peak in peaks:
    if peak[0] == "chrX":
        chrmNum = 9
    elif peak[0] == "chrXI":
        chrmNum = 10
    else:
        print("ERROR: Unexpected chromosome")
    seq = fasta_sequences[chrmNum][peak[1]:peak[2]]
    if len(seq) == seq_len:
        seq_peaks.append(seq)
        
betseq_original = "GATCTACACTCTTTCCCTACACGACGCTCTTCCGATCTTNNGTATCACGAGNCAATACACTGTTATCNNNNNCACGTGNNNNNCTACTCGTTCGGTTANCAGGAGAGCTNNAGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"

def fill_into_center(seq, insert):
    flank = int((len(seq)-len(insert))/2.0)
    new_seq = seq[:flank]
    for nuc_id, nuc in enumerate(insert):
        if nuc == 'N':
            new_seq += seq[flank+nuc_id]
        else:
            new_seq += nuc
    new_seq += seq[flank+len(insert):]
    return new_seq

from deeplift.dinuc_shuffle import dinuc_shuffle
candidates = []
for seq in seq_peaks:
    background = dinuc_shuffle(seq)
    candidates.append(fill_into_center(seq, betseq_original))
preds = model.predict([getOneHot(candidates), np.zeros((len(candidates),)), np.zeros((len(candidates),out_pred_len,2))])
count_preds = np.mean(preds[0], axis=1)
for idx, task in enumerate(task_names):
    count_preds += np.mean(preds[idx], axis=1)

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

profile_model_counts_explainer = [shap.explainers.deep.TFDeepExplainer(
    ([model.input[0], model.input[1]],
     tf.reduce_sum(model.outputs[idx],axis=-1)),
    shuffle_several_times,
    combine_mult_and_diffref=combine_mult_and_diffref) for idx in range(len(task_names))]

profile_model_profile_explainer = []
for idx in range(len(task_names), 2*len(task_names)):
    #See Google slide deck for explanations
    #We meannorm as per section titled "Adjustments for Softmax Layers"
    # in the DeepLIFT paper
    meannormed_logits = (
        model.outputs[idx]-
        tf.reduce_mean(model.outputs[idx],axis=1)[:,None,:])
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
    profile_model_profile_explainer.append(shap.explainers.deep.TFDeepExplainer(
        ([model.input[0], model.input[2]],
         weightedsum_meannormed_logits),
        shuffle_several_times,
        combine_mult_and_diffref=combine_mult_and_diffref))

seqs = []
test_seqs = []
test_preds_logcount = []
test_biastrack_logcount = []
test_biastrack_profile = []
test_preds_profile = []
test_labels_logcount = []
test_labels_profile = []
background = candidates[np.argmin(count_preds)]
for flank_id, flank in enumerate(flanks):
    seqs.append(fill_into_center(background, flank[:5] + "CACGTG" + flank[5:]))
    if flank_id % 100 == 0: 
        test_seqs.append(getOneHot(seqs))
        test_preds = model.predict([getOneHot(seqs), np.zeros((len(seqs),)),np.zeros((len(seqs),out_pred_len,2))])
        test_preds_logcount.append(test_preds[:len(task_names)])
        test_preds_profile.append(test_preds[len(task_names):])
        seqs = []
    if flank_id % 100000 == 0:
        print("completed processing ", flank_id, " flanks")
if len(seqs) != 0:
    test_seqs.append(getOneHot(seqs))
    test_preds = model.predict([getOneHot(seqs), np.zeros((len(seqs),)),np.zeros((len(seqs),out_pred_len,2))])
    test_preds_logcount.append(test_preds[:len(task_names)])
    test_preds_profile.append(test_preds[len(task_names):])
test_seqs = np.concatenate(test_seqs,axis=0)
test_preds_logcount = np.concatenate(test_preds_logcount, axis=1)
test_preds_profile = np.concatenate(test_preds_profile, axis=1)
test_post_counts_hypimps = [profile_model_counts_explainer[idx].shap_values(
    [test_seqs, np.zeros((len(test_seqs), 1))],
    progress_message=10)[0] for idx in range(len(task_names))]
test_post_profile_hypimps = [profile_model_profile_explainer[idx].shap_values(
    [test_seqs, np.zeros((len(test_seqs), out_pred_len, 2))],
    progress_message=10) [0] for idx in range(len(task_names))]
test_post_counts_hypimps = np.array(test_post_counts_hypimps)
test_post_profile_hypimps = np.array(test_post_profile_hypimps)
test_post_counts_actualimps = test_post_counts_hypimps*test_seqs
test_post_profile_actualimps = test_post_profile_hypimps*test_seqs
np.save(options.output_dir+'flanks_post_counts_hypimps.npy', test_post_counts_hypimps)
np.save(options.output_dir+'flanks_post_profile_hypimps.npy', test_post_profile_hypimps) 
np.save(options.output_dir+'flanks_post_counts_actualimps.npy', test_post_counts_actualimps) 
np.save(options.output_dir+'flanks_post_profile_actualimps.npy', test_post_profile_actualimps) 
np.save(options.output_dir+'flanks_preds_profile.npy', test_preds_profile) 
np.save(options.output_dir+'flanks_preds_logcount.npy', test_preds_logcount) 
np.save(options.output_dir+'flanks_seqs.npy', test_seqs)