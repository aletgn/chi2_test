import sys
sys.path.append("../src/")
                
from pygof.test import chi2test
from pygof.binning import re_bin
from pygof.util import random_variable, inspect_sample

from scipy.stats import norm

import numpy as np
np.random.seed(1)
num_bins = 50

# sample size
sample_size = 5000

# build the ``reference'' random variable
rv = random_variable(norm, loc=0, scale=1)
sample = rv.rvs(size = sample_size)

# inspect_sample(sample, n_bins=num_bins, rv=rv, density=True)

merged_counts, merged_edges, merged_edges_plot = re_bin(sample, n_bins=num_bins, th=5)

chi2test(merged_counts, merged_edges, rv, 10, est_params=True)

inspect_sample(sample, n_bins=num_bins, density=False, new_counts=merged_counts, new_edges=merged_edges_plot)