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
import codecs
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
        
from deeplift.dinuc_shuffle import dinuc_shuffle
num_samples = 100
flankToDeltaLogCount = {}
for flank_id, flank in enumerate(flanks):
    if flank_id % 100000 == 0:
        print("completed processing ", flank_id, " flanks")
    pre_seqs = []
    post_seqs = []
    insert = flank[:5] + "CACGTG" + flank[5:]
    insert_len = len(insert)
    start = int((seq_len/2)-(insert_len/2))
    indices = np.random.choice(len(seq_peaks), num_samples, replace=False)
    for idx in indices:
        pre_seq = dinuc_shuffle(seq_peaks[idx])
        post_seq = pre_seq[:start] + insert + pre_seq[start+insert_len:]
        pre_seqs.append(pre_seq)
        post_seqs.append(post_seq)
    pre = model.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    post = model.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    
    flankToDeltaLogCount[flank] = [pre[0].tolist(),
                                   #pre[1].tolist(),
                                   post[0].tolist(),
                                   #post[1].tolist()
                                  ]
json.dump(flankToDeltaLogCount,
          codecs.open(options.output_json, 'w', encoding='utf-8'),
          separators=(',', ':'),
          sort_keys=True, indent=4)

## In order to "unjsonify" the array use:
# obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# a_new = np.array(b_new)