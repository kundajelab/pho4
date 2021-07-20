import os
import sys
import pickle
import numpy as np
from math import exp
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from vizsequence.viz_sequence import plot_weights_given_ax
from scipy.special import softmax
import keras
import keras.losses
from keras.models import Model, Sequential, load_model
from keras import backend as K
import numpy.random as rng
import seaborn as sns
from collections import OrderedDict
from basepair.losses import twochannel_multinomial_nll
import modisco
import modisco.tfmodisco_workflow.workflow
import h5py
import modisco.util
from collections import Counter
from modisco.visualization import viz_sequence
import modisco.affinitymat.core
import modisco.cluster.phenograph.core
import modisco.cluster.phenograph.cluster
import modisco.cluster.core
import modisco.aggregator

os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

all_post_counts_hypimps = np.load('data/imp-scores/120min_pho4_nexus/post_counts_hypimps.npy')
all_post_profile_hypimps = np.load('data/imp-scores/120min_pho4_nexus/post_profile_hypimps.npy') 
all_post_counts_actualimps = np.load('data/imp-scores/120min_pho4_nexus/post_counts_actualimps.npy') 
all_post_profile_actualimps = np.load('data/imp-scores/120min_pho4_nexus/post_profile_actualimps.npy') 
all_labels_profile = np.load('data/imp-scores/120min_pho4_nexus/labels_profile.npy') 
all_labels_logcount = np.load('data/imp-scores/120min_pho4_nexus/labels_logcount.npy') 
all_preds_profile = np.load('data/imp-scores/120min_pho4_nexus/preds_profile.npy') 
all_preds_logcount = np.load('data/imp-scores/120min_pho4_nexus/preds_logcount.npy')
all_seqs = np.load('data/imp-scores/120min_pho4_nexus/seqs.npy') 

print(all_post_counts_hypimps.shape)
print(all_post_profile_hypimps.shape)
print(all_post_counts_actualimps.shape)
print(all_post_profile_actualimps.shape)
print(all_labels_profile.shape)
print(all_labels_logcount.shape)
print(all_preds_profile.shape)
print(all_preds_logcount.shape)
print(all_seqs.shape)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

tfname = "task0"
grad_wrt = "task0"
counts_attributions = all_post_counts_actualimps
counts_hyp_attributions = all_post_counts_hypimps
counts_tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
    sliding_window_size=15,
    flank_size=5,
    target_seqlet_fdr=0.15,
    seqlets_to_patterns_factory=
        modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            trim_to_window_size=15,
            initial_flank_to_add=5,
            kmer_len=5, num_gaps=1,
            num_mismatches=0,
            final_min_cluster_size=60))(
    task_names=[tfname],
    contrib_scores={grad_wrt: counts_attributions},
    hypothetical_contribs={grad_wrt: counts_hyp_attributions},
    one_hot=all_seqs)
counts_grp = h5py.File("data/modisco/120min_pho4_nexus/counts_results.hdf5")
counts_tfmodisco_results.save_hdf5(counts_grp)
    
profile_attributions = all_post_profile_actualimps
profile_hyp_attributions = all_post_profile_hypimps
profile_tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
    sliding_window_size=15,
    flank_size=5,
    target_seqlet_fdr=0.15,
    seqlets_to_patterns_factory=
        modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            trim_to_window_size=15,
            initial_flank_to_add=5,
            kmer_len=5, num_gaps=1,
            num_mismatches=0,
            final_min_cluster_size=60))(
    task_names=[tfname],
    contrib_scores={grad_wrt: profile_attributions},
    hypothetical_contribs={grad_wrt: profile_hyp_attributions},
    one_hot=all_seqs)
profile_grp = h5py.File("data/modisco/120min_pho4_nexus/profile_results.hdf5")
profile_tfmodisco_results.save_hdf5(profile_grp)