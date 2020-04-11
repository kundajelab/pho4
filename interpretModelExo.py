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
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import optparse
import os

parser = optparse.OptionParser()

parser.add_option('--gpus',
    action="store", dest="gpus",
    help="which gpus to use", default=None)
parser.add_option('--assay',
    action="store", dest="assay",
    help="which control to use", default=None)
parser.add_option('--task_names',
    action="store", dest="task_names",
    help="what are the tasks", default=None)
parser.add_option('--out_pred_len',
    action="store", dest="out_pred_len",
    help="length of predicted profile", default=None)
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
pos_neg_smooth_log_counts =\
  coordstovals.bigwig.PosAndNegSmoothWindowCollapsedLogCounts(
        pos_strand_bigwig_path="/users/amr1/pho4/data/ctl_"+options.assay+"/"+options.assay+".pos_strand.bw",
        neg_strand_bigwig_path="/users/amr1/pho4/data/ctl_"+options.assay+"/"+options.assay+".neg_strand.bw",
        counts_mode_name="control_logcount",
        profile_mode_name="control_profile",
        center_size_to_use=out_pred_len,
        smoothing_windows=[1,50])

inputs_coordstovals = coordstovals.core.CoordsToValsJoiner(
    coordstovals_list=[
      coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
        genome_fasta_path="/users/amr1/pho4/data/genome/sacCer3.genome.fa",
        mode_name="sequence",
        center_size_to_use=seq_len),
      pos_neg_smooth_log_counts])

targets_coordstovals = coordstovals.core.CoordsToValsJoiner(
    coordstovals_list=[
      coordstovals.bigwig.PosAndNegSeparateLogCounts(
        counts_mode_name=task+".logcount",
        profile_mode_name=task+".profile",
        pos_strand_bigwig_path="/users/amr1/pho4/data/"+task+"_pbexo/basename_prefix.pooled.positive.bigwig",
        neg_strand_bigwig_path="/users/amr1/pho4/data/"+task+"_pbexo/basename_prefix.pooled.negative.bigwig",
        center_size_to_use=out_pred_len) for task in task_names])

keras_test_batch_generator = coordbased.core.KerasBatchGenerator(
  coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=options.peaks_bed,
            batch_size=64,
            shuffle_before_epoch=False, 
            seed=1234),
  inputs_coordstovals=inputs_coordstovals,
  targets_coordstovals=targets_coordstovals)

import numpy as np

def extend_generator(generator):
    samp_inputs, samp_targets = generator[0]
    concat_inputs = OrderedDict([(key, []) for key in samp_inputs.keys()])
    concat_targets = OrderedDict([(key, []) for key in samp_targets.keys()])
    for batch_idx in range(len(generator)):
        batch_inputs, batch_targets = generator[batch_idx]
        for key in batch_inputs:
            concat_inputs[key].extend(batch_inputs[key])
        for key in batch_targets:
            concat_targets[key].extend(batch_targets[key])
    for key in concat_inputs:
        concat_inputs[key] = np.array(concat_inputs[key])
    for key in concat_targets:
        concat_targets[key] = np.array(concat_targets[key])
    return (concat_inputs, concat_targets)

test_inputs, test_targets = extend_generator(keras_test_batch_generator)

for idx in range(len(task_names)):
    print(model.outputs[idx])
    print(model.outputs[idx+len(task_names)])
    print(task_names[idx])
    
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

test_seqs = []
test_preds_logcount = []
test_biastrack_logcount = []
test_biastrack_profile = []
test_preds_profile = []
test_labels_logcount = []
test_labels_profile = []

for batch_idx in range(len(keras_test_batch_generator)):
    batch_inputs, batch_labels = keras_test_batch_generator[batch_idx]
    test_seqs.append(batch_inputs['sequence']) 
    test_biastrack_logcount.append(batch_inputs['control_logcount'])
    test_biastrack_profile.append(batch_inputs['control_profile'])    
    test_preds = model.predict(batch_inputs)
    test_preds_logcount.append(test_preds[:len(task_names)])
    test_preds_profile.append(test_preds[len(task_names):])
    test_labels_logcount.append([batch_labels[task+'.logcount'] for task in task_names])
    test_labels_profile.append([batch_labels[task+'.profile'] for task in task_names])
test_seqs = np.concatenate(test_seqs,axis=0)
test_biastrack_logcount = np.concatenate(test_biastrack_logcount, axis=0)
test_biastrack_profile = np.concatenate(test_biastrack_profile,axis=0)
test_preds_logcount = np.concatenate(test_preds_logcount, axis=1)
test_preds_profile = np.concatenate(test_preds_profile, axis=1)
test_labels_logcount = np.concatenate(test_labels_logcount, axis=1)
test_labels_profile = np.concatenate(test_labels_profile, axis=1)

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

np.save(options.output_dir+'post_counts_hypimps.npy', test_post_counts_hypimps)
np.save(options.output_dir+'post_profile_hypimps.npy', test_post_profile_hypimps) 
np.save(options.output_dir+'post_counts_actualimps.npy', test_post_counts_actualimps) 
np.save(options.output_dir+'post_profile_actualimps.npy', test_post_profile_actualimps) 
np.save(options.output_dir+'labels_profile.npy', test_labels_profile) 
np.save(options.output_dir+'labels_logcount.npy', test_labels_logcount) 
np.save(options.output_dir+'preds_profile.npy', test_preds_profile) 
np.save(options.output_dir+'biastrack_profile.npy', test_biastrack_profile) 
np.save(options.output_dir+'biastrack_logcount.npy', test_biastrack_logcount) 
np.save(options.output_dir+'preds_logcount.npy', test_preds_logcount) 
np.save(options.output_dir+'seqs.npy', test_seqs) 