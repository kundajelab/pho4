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
from math import log
import json
import codecs
import os
import gzip
import optparse

parser = optparse.OptionParser()

parser.add_option('--gpus',
    action="store", dest="gpus",
    help="which gpus to use", default=None)
parser.add_option('--out_pred_len',
    action="store", dest="out_pred_len",
    help="what is the length of output", default=None)
parser.add_option('--peaks_bed',
    action="store", dest="peaks_bed",
    help="where are the peaks", default=None)
parser.add_option('--model',
    action="store", dest="model",
    help="where is the model", default=None)
parser.add_option('--test_chrms',
    action="store", dest="test_chrms",
    help="what are the test_chrms", default=None)
parser.add_option('--output_json',
    action="store", dest="output_json",
    help="where to store end result", default=None)

options, args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=options.gpus

seqs = []
firstLine = True
with gzip.open("data/experimental/competition/GSM4980364_8uMPho4_GST_alldata.txt.gz", 'rt') as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        curr = line.strip().split('\t')[4]
        if curr == "_": continue
        if curr[36:] != "GTCTTGATTCGCTTGACGCTGCTG": print("Exception: ", curr)
        seqs.append(curr[:36])
        #val = log(float(line.strip().split('\t')[-1]))

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
peaks = []
test_chrms = options.test_chrms.split(',')
print(test_chrms)
with open(options.peaks_bed) as inp:
    for line in inp:
        chrm = line.strip().split('\t')[0]
        if chrm not in test_chrms:
            continue
        pStart = int(line.strip().split('\t')[1])
        pEnd = int(line.strip().split('\t')[2])
        summit = pStart + int((pEnd-pStart)/2)    #int(line.strip().split('\t')[-1])  
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
genome_object = Fasta("/users/amr1/pho4/data/genome/saccer/sacCer3.genome.fa")

chrom_sizes = readChromSizes("/users/amr1/pho4/data/genome/saccer/sacCer3.chrom.sizes")
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

chrmsToChrmNums = {'chrI': 0, 'chrII': 1, 'chrIII': 2, 'chrIV': 3,
                   'chrV': 4, 'chrVI': 5, 'chrVII': 6, 'chrVIII': 7,
                   'chrIX': 8, 'chrX': 9, 'chrXI': 10, 'chrXII': 11,
                   'chrXIII': 12, 'chrXIV': 13, 'chrXV': 14, 'chrXVI': 15}

seq_peaks = []
for peak in peaks:
    seq = fasta_sequences[chrmsToChrmNums[peak[0]]][peak[1]:peak[2]]
    if len(seq) == seq_len:
        seq_peaks.append(seq)
        
from deeplift.dinuc_shuffle import dinuc_shuffle
num_samples = 100
seqsToDeltaLogCount = {}
for seq_id, seq in enumerate(seqs):
    if seq_id % 10000 == 0:
        print("completed processing ", seq_id, " seqs")
    pre_seqs = []
    post_seqs = []
    insert_len = len(seq)
    start = int((seq_len/2)-(insert_len/2))
    indices = np.random.choice(len(seq_peaks), num_samples, replace=False)
    for idx in indices:
        pre_seq = dinuc_shuffle(seq_peaks[idx])
        post_seq = pre_seq[:start] + seq + pre_seq[start+insert_len:]
        pre_seqs.append(pre_seq)
        post_seqs.append(post_seq)
    if 'nexus' in options.peaks_bed:
        pre = model.predict(getOneHot(pre_seqs))
        post = model.predict(getOneHot(post_seqs))
    else:
        pre = model.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
        post = model.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    
    seqsToDeltaLogCount[seq] = [pre[0].tolist(), post[0].tolist()]
json.dump(seqsToDeltaLogCount,
          codecs.open(options.output_json, 'w', encoding='utf-8'),
          separators=(',', ':'),
          sort_keys=True, indent=4)

## In order to "unjsonify" the array use:
# obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# a_new = np.array(b_new)