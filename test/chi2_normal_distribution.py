import sys
sys.path.append("../src/")
                
from pygof.chi2_test import build_rv, generate_sample, recompute_histogram, do_chi2_test

# numeber of bins
num_samples = 50

# sample size
sample_size = 5000

# build the ``reference'' random variable
rv = build_rv(mu=0, std=1, family='norm')

# sample
x = generate_sample(rv, size = sample_size, n_bins=num_samples, plot=True)

# merge bins and recompute histogram
merged_counts, merged_bins = recompute_histogram(x, n_bins=num_samples, th=5, plot=True)

# do chi^2 test
do_chi2_test(merged_counts, merged_bins, rv, 10, est_params=True)