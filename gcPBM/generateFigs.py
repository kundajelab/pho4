import numpy as np
import pandas as pd
import json
import os
import gzip
from math import log
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import optparse

parser = optparse.OptionParser()

parser.add_option('--library',
    action="store", dest="library",
    help="where seqs at", default=None)
parser.add_option('--tf',
    action="store", dest="tf",
    help="tf", default=None)
parser.add_option('--column',
    action="store", dest="column",
    help="column", default=None)

options, args = parser.parse_args()

dfs = pd.read_excel(options.library)
all_xvals = dfs[options.column]
yvals = np.load('preds/'+options.tf+'_yvals.npy')
yvals_counts = np.load('preds/'+options.tf+'_yvals_counts.npy')

xy = np.vstack([all_xvals,yvals])
z = gaussian_kde(xy)(xy)
smallFont = {'size' : 10}
plt.rc('font', **smallFont)
fig, ax = plt.subplots()
ax.scatter(all_xvals, yvals, c=z, edgecolor='', alpha=0.5)
plt.xlabel("Log gcPBM Signal")
plt.ylabel("Delta Log Counts")
plt.title("spearman: "+ \
          str(spearmanr(all_xvals, yvals)[0]) + ", pearson: "+ str(pearsonr(all_xvals, yvals)[0]))
fig.savefig('figs/'+options.column+'_loglog.png', dpi=fig.dpi)

xy = np.vstack([all_xvals,yvals_counts])
z = gaussian_kde(xy)(xy)
smallFont = {'size' : 10}
plt.rc('font', **smallFont)
fig, ax = plt.subplots()
ax.scatter(all_xvals, yvals_counts, c=z, edgecolor='', alpha=0.5)
plt.xlabel("Log gcPBM Signal")
plt.ylabel("Delta Counts")
plt.title("spearman: "+ \
          str(spearmanr(all_xvals, yvals_counts)[0]) + ", pearson: "+ str(pearsonr(all_xvals, yvals_counts)[0]))
fig.savefig('figs/'+options.column+'_logpbm.png', dpi=fig.dpi)

xvals_exp = np.exp(all_xvals)
xy = np.vstack([xvals_exp,yvals_counts])
z = gaussian_kde(xy)(xy)
smallFont = {'size' : 10}
plt.rc('font', **smallFont)
fig, ax = plt.subplots()
ax.scatter(xvals_exp, yvals_counts, c=z, edgecolor='', alpha=0.5)
plt.xlabel("gcPBM Signal")
plt.ylabel("Delta Counts")
plt.title("spearman: "+ \
          str(spearmanr(xvals_exp, yvals_counts)[0]) + ", pearson: "+ str(pearsonr(xvals_exp, yvals_counts)[0]))
fig.savefig('figs/'+options.column+'_nologs.png', dpi=fig.dpi)

xy = np.vstack([xvals_exp,yvals])
z = gaussian_kde(xy)(xy)
smallFont = {'size' : 10}
plt.rc('font', **smallFont)
fig, ax = plt.subplots()
ax.scatter(xvals_exp, yvals, c=z, edgecolor='', alpha=0.5)
plt.xlabel("gcPBM Signal")
plt.ylabel("Delta Log Counts")
plt.title("spearman: "+ \
          str(spearmanr(xvals_exp, yvals)[0]) + ", pearson: "+ str(pearsonr(xvals_exp, yvals)[0]))
fig.savefig('figs/'+options.column+'_logcounts.png', dpi=fig.dpi)