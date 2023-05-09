# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 09:07:55 2023

@author: AT
"""
import numpy as np; np.random.seed(1)
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import chi2


def unravel_bins(bins):
    u_bins = []
    
    for b in range(0, len(bins)-1):
        u_bins.append([bins[b], bins[b+1]])
        
    return np.array(u_bins, dtype = np.float64)


def merge_bins(counts, bins, th = 5, flip = False):
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


def compute_chi2(bins, counts, mu = 0, std = 1):
    # compute theretical frequencies for all bins
    gauss = norm(loc = 0, scale = 1)
    th = counts.sum()*np.array([gauss.cdf(k[1]) - gauss.cdf(k[0]) for k in bins])
    
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


# This function will be deprecated
# def gauss(x, mu, std):
#     return np.exp(-0.5*(((x-mu)/std)**2))/((np.pi*2*(std**2))**0.5)



if __name__ == '__main__':
    # numeber of bins
    b = 50
    
    # sample size
    ss = 5000
    
    x = norm(loc = 0, scale = 1).rvs(size = ss)
    plt.figure()
    plt.hist(x, bins = b, histtype='step', color = 'red')

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
    th = 10
    
    # on the left
    sx_new_counts, sx_new_bins = merge_bins(sx_counts, sx_bins, th = th)
    # on the right -- watch out: flip the array
    dx_new_counts, dx_new_bins = merge_bins(dx_counts, dx_bins, th = th, flip = True)
    
    # join new counts and bins
    new_counts = np.concatenate([sx_new_counts, dx_new_counts])
    new_bins = np.concatenate([sx_new_bins, dx_new_bins])
    
    # in order to plot take the edges of the distribution
    new_edges = np.array([new_bins[0,0]] + list(new_bins[:,1]))
    plt.stairs(new_counts, new_edges, fill=True, color = 'blue')
    
    # display gaussian distribution as well
    plt.figure(dpi = 300)
    plt.hist(x, bins = b, histtype='step', color = 'red', density=True)
    x_coord = np.linspace(x_bins.min(), x_bins.max(), 1000)
    plt.plot(x_coord, norm(loc = 0, scale = 1).pdf(x_coord), c='k')
    
    # logging 
    print(counts)
    print(new_counts)
    print()
    
    # chi^2 from data
    data_chi_2 = compute_chi2(new_bins, new_counts)
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