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
import optparse
import os
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde

os.environ["CUDA_VISIBLE_DEVICES"]="4,5"

parser = optparse.OptionParser()

parser.add_option('--val',
    action="store", dest="val",
    help="which one", default=None)

options, args = parser.parse_args()

key = options.val
model_path = "/users/amr1/pho4/data/models/er_"+key+".h5"
bed_path = "/users/amr1/pho4/data/er/"+key+"/1k_around_summits.bed.gz"

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

from keras.models import load_model
from keras.utils import CustomObjectScope

with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
    model = load_model(model_path)

seq_len = 1346
out_pred_len = 1000
inputs_coordstovals = coordstovals.core.CoordsToValsJoiner(
    coordstovals_list=[
      coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
        genome_fasta_path="/users/amr1/pho4/data/genome/hg19/male.hg19.fa",
        mode_name="sequence",
        center_size_to_use=seq_len),
      coordstovals.bigwig.PosAndNegSmoothWindowCollapsedLogCounts(
        pos_strand_bigwig_path="/users/amr1/pho4/data/er/ctl/control_pos_strand.bw",
        neg_strand_bigwig_path="/users/amr1/pho4/data/er/ctl/control_neg_strand.bw",
        counts_mode_name="control_logcount",
        profile_mode_name="control_profile",
        center_size_to_use=out_pred_len,
        smoothing_windows=[1,50])])

targets_coordstovals = coordstovals.bigwig.PosAndNegSeparateLogCounts(
    pos_strand_bigwig_path="/users/amr1/pho4/data/er/"+key+"/pos.bigwig",
    neg_strand_bigwig_path="/users/amr1/pho4/data/er/"+key+"/neg.bigwig",
    counts_mode_name="task0_logcount",
    profile_mode_name="task0_profile",
    center_size_to_use=out_pred_len)

keras_data_batch_generator = coordbased.core.KerasBatchGenerator(
  coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=bed_path,
            batch_size=128,
            shuffle_before_epoch=False, 
            seed=1234),
  inputs_coordstovals=inputs_coordstovals,
  targets_coordstovals=targets_coordstovals)

import shap
from deeplift.dinuc_shuffle import dinuc_shuffle

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

test_preds_logcount = []
test_biastrack_logcount = []
test_biastrack_profile = []
test_seqs = []
test_preds_profile = []
test_labels_logcount = []
test_labels_profile = []
for batch_idx in range(len(keras_data_batch_generator)):
    batch_inputs, batch_labels = keras_data_batch_generator[batch_idx]
    test_seqs.append(batch_inputs['sequence'])
    test_biastrack_logcount.append(batch_inputs['control_logcount'])
    test_biastrack_profile.append(batch_inputs['control_profile'])
    test_preds = model.predict(batch_inputs)
    test_preds_logcount.append(test_preds[0])
    test_preds_profile.append(test_preds[1])
    test_labels_logcount.append(batch_labels['task0_logcount'])
    test_labels_profile.append(batch_labels['task0_profile'])
test_biastrack_logcount = np.concatenate(test_biastrack_logcount, axis=0)
test_biastrack_profile = np.concatenate(test_biastrack_profile,axis=0)
test_seqs = np.concatenate(test_seqs,axis=0)
test_preds_logcount = np.concatenate(test_preds_logcount, axis=0)
test_preds_profile = np.concatenate(test_preds_profile, axis=0)
test_labels_logcount = np.concatenate(test_labels_logcount, axis=0)
test_labels_profile = np.concatenate(test_labels_profile, axis=0)

#The shap scores
test_post_counts_hypimps,_ = profile_model_counts_explainer.shap_values(
    [test_seqs, np.zeros((len(test_seqs), 1))],
    progress_message=10)
test_post_profile_hypimps,_ = profile_model_profile_explainer.shap_values(
    [test_seqs, np.zeros((len(test_seqs), out_pred_len, 2))],
    progress_message=10)
test_post_counts_hypimps = np.array(test_post_counts_hypimps)
test_post_profile_hypimps = np.array(test_post_profile_hypimps)
test_post_counts_actualimps = test_post_counts_hypimps*test_seqs
test_post_profile_actualimps = test_post_profile_hypimps*test_seqs

np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/post_counts_hypimps.npy', test_post_counts_hypimps)
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/post_profile_hypimps.npy', test_post_profile_hypimps) 
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/post_counts_actualimps.npy', test_post_counts_actualimps) 
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/post_profile_actualimps.npy', test_post_profile_actualimps) 
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/labels_profile.npy', test_labels_profile) 
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/labels_logcount.npy', test_labels_logcount) 
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/preds_profile.npy', test_preds_profile) 
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/biastrack_profile.npy', test_biastrack_profile) 
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/biastrack_logcount.npy', test_biastrack_logcount) 
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/preds_logcount.npy', test_preds_logcount) 
np.save('/users/amr1/pho4/data/imp-scores/er/'+key+'/seqs.npy', test_seqs) 