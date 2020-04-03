import keras_genomics
from keras_genomics.layers.convolutional import RevCompConv1D
import keras
import keras.layers as kl
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
import os
import optparse

parser = optparse.OptionParser()

parser.add_option('--gpus',
    action="store", dest="gpus",
    help="which gpus to use", default=None)
parser.add_option('--flanks_csv',
    action="store", dest="flanks_csv",
    help="where are the flanks", default=None)
parser.add_option('--peaks_bed',
    action="store", dest="peaks_bed",
    help="where are the peaks", default=None)
parser.add_option('--model',
    action="store", dest="model",
    help="where is the model", default=None)
parser.add_option('--output_json',
    action="store", dest="output_json",
    help="where to store end result", default=None)

options, args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=options.gpus
flanks = []
firstLine = True
with open(options.flanks_csv) as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flanks.append(line.strip().split(',')[0])

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
out_pred_len = 200
peaks = []
test_chrms = ["chrX", "chrXI"]
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
count_preds += np.mean(preds[1], axis=1)

seqs = []
preds = np.array([])
background = candidates[np.argmin(count_preds)]
preds_task2 = np.array([])
for flank_id, flank in enumerate(flanks):
    seqs.append(fill_into_center(background, flank[:5] + "CACGTG" + flank[5:]))
    if flank_id % 100 == 0: 
        if preds.shape[0] == 0:
            preds = model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[0]
            preds_task2 = model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[1]
        else:
            preds = np.vstack((preds,
                           model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[0]))
            preds_task2 = np.vstack((preds,
                           model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[1]))
        seqs = []
    if flank_id % 100000 == 0:
        print("completed processing ", flank_id, " flanks")
if len(seqs) != 0:
    preds = np.vstack((preds,
                       model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                      np.zeros((len(seqs),out_pred_len,2))])[0]))
    preds_task2 = np.vstack((preds,
                           model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[1]))
preds = np.mean(preds, axis=1)
preds_task2 = np.mean(preds_task2, axis=1)
flankToLogCount = {}
for flank_id, flank in enumerate(flanks):
    flankToLogCount[flank] = (str(preds[flank_id]), str(preds_task2[flank_id]))
with open(options.output_json+"_fixed_min.json", 'w') as fp:
    json.dump(flankToLogCount, fp)
    
seqs = []
preds = np.array([])
background = candidates[np.argsort(count_preds)[len(count_preds)//2]]
preds_task2 = np.array([])
for flank_id, flank in enumerate(flanks):
    seqs.append(fill_into_center(background, flank[:5] + "CACGTG" + flank[5:]))
    if flank_id % 100 == 0: 
        if preds.shape[0] == 0:
            preds = model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[0]
            preds_task2 = model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[1]
        else:
            preds = np.vstack((preds,
                           model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[0]))
            preds_task2 = np.vstack((preds,
                           model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[1]))
        seqs = []
    if flank_id % 100000 == 0:
        print("completed processing ", flank_id, " flanks")
if len(seqs) != 0:
    preds = np.vstack((preds,
                       model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                      np.zeros((len(seqs),out_pred_len,2))])[0]))
    preds_task2 = np.vstack((preds,
                           model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[1]))
preds = np.mean(preds, axis=1)
preds_task2 = np.mean(preds_task2, axis=1)
flankToLogCount = {}
for flank_id, flank in enumerate(flanks):
    flankToLogCount[flank] = (str(preds[flank_id]), str(preds_task2[flank_id]))
with open(options.output_json+"_fixed_median.json", 'w') as fp:
    json.dump(flankToLogCount, fp)
    
seqs = []
preds = np.array([])
background = candidates[np.argmax(count_preds)]
preds_task2 = np.array([])
for flank_id, flank in enumerate(flanks):
    seqs.append(fill_into_center(background, flank[:5] + "CACGTG" + flank[5:]))
    if flank_id % 100 == 0: 
        if preds.shape[0] == 0:
            preds = model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[0]
            preds_task2 = model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[1]
        else:
            preds = np.vstack((preds,
                           model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[0]))
            preds_task2 = np.vstack((preds,
                           model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[1]))
        seqs = []
    if flank_id % 100000 == 0:
        print("completed processing ", flank_id, " flanks")
if len(seqs) != 0:
    preds = np.vstack((preds,
                       model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                      np.zeros((len(seqs),out_pred_len,2))])[0]))
    preds_task2 = np.vstack((preds,
                           model.predict([getOneHot(seqs), np.zeros((len(seqs),)),
                                          np.zeros((len(seqs),out_pred_len,2))])[1]))
preds = np.mean(preds, axis=1)
preds_task2 = np.mean(preds_task2, axis=1)
flankToLogCount = {}
for flank_id, flank in enumerate(flanks):
    flankToLogCount[flank] = (str(preds[flank_id]), str(preds_task2[flank_id]))
with open(options.output_json+"_fixed_max.json", 'w') as fp:
    json.dump(flankToLogCount, fp)