# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:07:55 2023

@author: AT
"""
import numpy as np; np.random.seed(1)
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import chi2


def unravel_bins(bins):
    """
    

    Parameters
    ----------
    bins : numpy array
        bins obtained by the application of np.histogram to the original sample
        bins has this configuration:
            
            edge_0 edge_1 edge_2 edge_3 ... edge_N
            
        this function gives:
            edge_0 edge_1
            
            edge_1 edge_2
            
            edge_2 edge_3
            
            ...
            
            edge_N-1 edge_N
            
            

    Returns
    -------
    numpy array
        see the transformation above.

    """
    u_bins = []
    
    for b in range(0, len(bins)-1):
        u_bins.append([bins[b], bins[b+1]])
        
    return np.array(u_bins, dtype = np.float64)


def merge_bins(counts, bins, th = 5, flip = False):
    """
    

    Parameters
    ----------
    counts : np array
        Array of the observed frequency obtained by np.histogram 
    bins : np array
        Array of the observed frequency obtained by np.histogram having
        the structure given by unravel bins
    th : INT
        Minimum number of samples per bin. The default is 5.
    flip : BOOL
        Flag to indicate whether the processed array is on the left of the mode
        of the sample histogram. The default is False (left). True means ``on
        the right''.

    Returns
    -------
    np array
        array of merged bins whose minumum number of sample is => 5.

    """
    # flip vector if necessary
    if flip:
        counts = np.flip(counts)
        bins = np.flip(bins)
    
    
    # new vectors
    new_counts = []
    new_bins = []

    # global index
    k = 0
    
    # auxiliary index
    h = 0
    
    # outer loop over count vector
    while k < counts.shape[0]:
        curr_count = counts[k]
        # print('--- ', curr_count)
        
        h = k
        # inner loop when bins are below th
        while curr_count < th:
            h += 1
            curr_count += counts[h]
            # print(curr_count, h)
        
        # append results
        new_bins.append([bins[k][0], bins[h][1]])
        new_counts.append(curr_count)
        
        # make the index jump forward when h-k > 0 
        k += h-k+1
    
    if flip:
        new_counts = np.flip(new_counts)
        new_bins = np.flip(new_bins)
        
    return np.array(new_counts), np.array(new_bins)
    # return new_counts, new_bins


def compute_chi2(bins, counts, rand_var):
    # compute theretical frequencies for all bins
    th = counts.sum()*np.array([rand_var.cdf(k[1]) - rand_var.cdf(k[0]) for k in bins])
    return (((counts - th)**2)/th).sum()


def my_chi2(signif, bins, est_params = 0):
    """
    

    Parameters
    ----------
    signif : int
        Significativity level. This should be entered as 5, 10, etc...
    bins : TYPE
        DESCRIPTION.
    est_params : int, optional
        DESCRIPTION. The default is 0. When the parameters of the underlying
        distribution are estimated from the data, the dof has to be decreased
        by the number of estimated parameters

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    dof = bins.shape[0] - 1 - est_params
    confidence = (100-signif)/100
    
    return chi2.ppf([confidence], dof)[0]


def build_rv(mu = 0, std = 1, s_log = 1, family = 'norm'):
    """
    This function allows one to create a random variable according to the
    desired distribution to be testes

    Parameters
    ----------
    mu : FLOAT, optional
        The mean of the normal distribution. The mean used to standardise the
        variable when using a lognormal. The default is 0.
    std : FLOAT, optional
        The std of the normal distribution. The std used to standardise the
        variable when using a lognormal. The default is 1.
    s_log : FLOAT, optional
        The scale of the lognormal distribution. The default is 1.
    family : STR
        Label for selecting the kind of distribution to be tested.
        The default is 'norm'.

    Raises
    ------
    NotImplementedError
        Raised when the selected distribution is wrong or does not exist.

    Returns
    -------
    scipy random variable

    """
    if family == 'norm':
        return norm(loc = mu, scale = std)
    elif family == 'lognorm':
        return lognorm(s = s_log, loc = mu, scale = std)
    else:
        raise NotImplementedError


def cat_sx_dx(sx, dx):
    if sx.shape[0] == 0:
        return dx
    elif dx.shape[0] == 0:
        return sx
    elif sx.shape[0] != 0 and dx.shape[0] != 0:
        return np.concatenate([sx, dx])
    else:
        raise Exception('There must be something wrong with sampling')


def generate_sample(rv, size = 1000, n_bins = 10, plot = False):
    """
    

    Parameters
    ----------
    rv : scipy random variable
        Generated via build_rv.
    size : INT
        The size of the sample. The default is 1000.
    n_bins : INT
        The number of bins. The default is 10.
    plot : BOOL
        Selector to decide whether display the sample. The default is False.

    Returns
    -------
    x : np array
        Sample.

    """
    
    # sample the random variable
    x = rv.rvs(size = ss)
    
    if plot:
        plt.figure(dpi = 300)
        plt.hist(x, bins = b, histtype='step', color = 'red')
        plt.title('Absolute Frequency')
        
        plt.figure(dpi = 300)
        plt.hist(x, bins = b, histtype='step', color = 'red', density=True)
        x_coord = np.linspace(x.min(), x.max(), 1000)
        plt.plot(x_coord, rv.pdf(x_coord), 'k')
        plt.title('Relative Frequency')
    return x


def recompute_histogram(sample, n_bins = 10, th = 5, plot=True, c_log=True):
    """
    This is an interface for processing the histogram of the given sample. This
    function recalls all the others to compute a full histogram whose bins have
    no less than th samples per bin

    Parameters
    ----------
    sample : np array
        DESCRIPTION.
    n_bins : INT
        Number of bins to generate. The default is 10.
    th : INT
        minumum number of bins per sample. The default is 5.
    plot : BOOL
        selector to decide whether display the histogram. The default is True.
    c_log : BOOL
        selector to decide whether log the original and merged numerosity.
        The default is True.

    Returns
    -------
    new_counts : TYPE
        DESCRIPTION.
    new_bins : TYPE
        DESCRIPTION.

    """
    
    # original histogram of the sample
    counts, x_bins = np.histogram(sample, bins=n_bins)
    
    # find a more convenient representation of bins
    bins = unravel_bins(x_bins)
    
    # find the index of the mode
    id_max = list(counts).index(counts.max())
    
    # split data wrt the mode of the distribution
    sx_bins = bins[None:id_max,:]
    dx_bins = bins[id_max:None,:]
    sx_counts = counts[None:id_max]
    dx_counts = counts[id_max:None]
    
    # on the left
    sx_new_counts, sx_new_bins = merge_bins(sx_counts, sx_bins, th = th)
    # on the right -- watch out: flip the array
    dx_new_counts, dx_new_bins = merge_bins(dx_counts, dx_bins, th = th, flip = True)
    
    # join new counts and bins
    new_counts = cat_sx_dx(sx_new_counts, dx_new_counts)
    new_bins = cat_sx_dx(sx_new_bins, dx_new_bins)
    
    # in order to plot take the edges of the distribution
    new_edges = np.array([new_bins[0,0]] + list(new_bins[:,1]))
    
    if plot:
        plt.figure(dpi = 300)
        plt.hist(sample, bins = b, histtype='step', color = 'red', label='Original')
        plt.stairs(new_counts, new_edges, fill=True, color = 'blue', label='Merged')
        plt.title('Absolute Frequency')
        plt.legend(loc='upper right')
        
    if c_log:
        print(f"\nMinimum bin size: {th:d}\n")
        print(f"Original numerosity per bin\n{counts}\n")
        print(f"Merged numerosity per bin\n{new_counts}\n")
        
    return new_counts, new_bins


def chi2_test(new_counts, new_bins, rand_var, signif=5, est_params=True):
    
    # chi^2 from data -- compute theoretical frequencies for all bins
    th = new_counts.sum()*np.array([rand_var.cdf(k[1]) - rand_var.cdf(k[0]) for k in new_bins])
    data_chi_2 = (((new_counts - th)**2)/th).sum()
    
    # specify wheter the parameters were estimated from the data
    if est_params:
        # get the number of parameters that can be estimated
        n_est_params = len(rand_var.kwds.keys())
    else:
        n_est_params = 0

    # chi^2 from theory
    # est_params parameters were estimated so dofs has to be decreased by est_params
    dof = new_bins.shape[0] - 1 - n_est_params
    confidence = (100-signif)/100
    fun_chi_2 = chi2.ppf([confidence], dof)[0]
    
    # logging
    print('*********************************')
    print(f"Number of samples {new_counts.sum():d}")
    print(f"Number of bins {len(new_bins):d}")
    print(f"Number of DoFs {dof:d}")
    print(f"Number of estimated parameters {n_est_params:d}")
    print(f"Chi2 from data = {data_chi_2:.2f}")
    print(f"Chi2 from function = {fun_chi_2:.2f}")
    # chi2 test
    if data_chi_2 < fun_chi_2:
        print("Chi2 test passed (data < fun)")
    else:
        print("Chi2 test failed (data >= fun)")
    print('*********************************')
    
if __name__ == '__main__':
    # family of the distribution
    f = 'norm'
    
    # numeber of bins
    b = 50
    
    # sample size
    ss = 5000
    
    # build the ``referece'' random variable
    rv = build_rv(mu=0, std=1, family=f)
    
    # sample
    x = generate_sample(rv, size = ss, n_bins=b, plot=False)
    
    # merge bins and recompute histogram
    merged_counts, merged_bins = recompute_histogram(x, n_bins=b, th=5, plot=False)
    
    # do chi^2 test
    chi2_test(merged_counts, merged_bins, rv, 10, est_params=False)
    
""" UNMODULARISED VERSION
    # take the histogram of x
    counts, x_bins = np.histogram(x, bins=b)
    
    # find a more convenient representation of bins
    bins = unravel_bins(x_bins)
    
    # find the index of the mode
    id_max = list(counts).index(counts.max())
    
    # split data wrt the mode of the distribution
    sx_bins = bins[None:id_max,:]
    dx_bins = bins[id_max:None,:]
    sx_counts = counts[None:id_max]
    dx_counts = counts[id_max:None]
    
    # merge bins and find a new histogram by threshold th
    th = 5
    
    # on the left
    sx_new_counts, sx_new_bins = merge_bins(sx_counts, sx_bins, th = th)
    # on the right -- watch out: flip the array
    dx_new_counts, dx_new_bins = merge_bins(dx_counts, dx_bins, th = th, flip = True)
    
    # join new counts and bins
    new_counts = cat_sx_dx(sx_new_counts, dx_new_counts)
    new_bins = cat_sx_dx(sx_new_bins, dx_new_bins)
    
    # in order to plot take the edges of the distribution
    new_edges = np.array([new_bins[0,0]] + list(new_bins[:,1]))
    plt.stairs(new_counts, new_edges, fill=True, color = 'blue')
    
    # display gaussian distribution as well
    plt.figure(dpi = 300)
    plt.hist(x, bins = b, histtype='step', color = 'red', density=True)
    x_coord = np.linspace(x_bins.min(), x_bins.max(), 1000)
    plt.plot(x_coord, rv.pdf(x_coord), c='k')
    
    plt.figure(dpi = 300)
    plt.hist(x, bins = b, histtype='step', color = 'red', label='Original')
    plt.stairs(new_counts, new_edges, fill=True, color = 'blue', label='Merged')
    plt.title('Absolute Frequency')
    
    # logging 
    print(counts)
    print(new_counts)
    print()
    
    print(f"Number of samples {ss:d}")
    # chi^2 from data
    # data_chi_2 = compute_chi2(new_bins, new_counts)
    data_chi_2 = compute_chi2(new_bins, new_counts, rv)
    print(f"Chi2 from data = {data_chi_2:.2f}")
    
    # chi^2 from the definition of chi^2
    #2 parameters were estimated so dofs has to be decreased by 2
    fun_chi_2 = my_chi2(10, new_bins, est_params=2)
    print(f"Chi2 from function = {fun_chi_2:.2f}")
    
    # chi2 test
    if data_chi_2 < fun_chi_2:
        print("Chi2 test passed (data < fun)")
    else:
        print("Chi2 test failed (data >= fun)")
"""

"""
new_counts = []
new_bins = []
th = 7

v = np.flip(dx_counts)
# v = sx_counts

# b = sx_bins
b = np.flip(dx_bins)

print(v)

k = 0
h = 0
while k < v.shape[0]:
    curr_count = v[k]
    print('--- ', curr_count)
    
    h = k
    while curr_count < th:
        h += 1
        curr_count += v[h]
        print(curr_count, h)
    
    new_bins.append([b[k][0], b[h][1]])
    new_counts.append(curr_count)
    k += h-k+1

new_counts = np.array(new_counts)
new_bins = np.array(new_bins)

print(v)
print(new_counts)
"""