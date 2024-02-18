import numpy as np
from typing import Tuple

def unravel_edges(edges: np.ndarray) -> np.ndarray:
    """
    Unravels edges obtained from the application of np.histogram to the original sample.

    Parameters
    ----------
    edges : numpy array
        Bins obtained by the application of np.histogram to the original sample.
        The 'edges' array has this configuration:

            edge_0 edge_1 edge_2 edge_3 ... edge_N

        This function transforms it into:

            edge_0 edge_1

            edge_1 edge_2

            edge_2 edge_3

            ...

            edge_N-1 edge_N

    Returns
    -------
    numpy array
        The transformed array representing pairs of consecutive edges.

    """
    u_edges = []
    
    for b in range(0, len(edges)-1):
        u_edges.append([edges[b], edges[b+1]])
        
    return np.array(u_edges, dtype = np.float64)


def merge_bins(counts: np.ndarray, edges: np.ndarray, th: int = 5, flip: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge bins with observed frequency counts below a specified threshold.

    Parameters
    ----------
    counts : np array
        Array of the observed frequency obtained by np.histogram.
    edges : np array
        Array of the observed frequency obtained by np.histogram having
        the structure given by unravel_bins.
    th : int, optional
        Minimum number of samples per bin. The default is 5.
    flip : bool, optional
        Flag to indicate whether the processed array is on the left of the mode
        of the sample histogram. The default is False (left). True means "on
        the right".

    Returns
    -------
    np array
        Array of merged bins whose minimum number of samples is >= 5.

    """
    # flip vector if necessary
    if flip:
        counts = np.flip(counts)
        edges = np.flip(edges)
    
    
    # new vectors
    new_counts = []
    new_edges = []

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
        new_edges.append([edges[k][0], edges[h][1]])
        new_counts.append(curr_count)
        
        # make the index jump forward when h-k > 0 
        k += h-k+1
    
    if flip:
        new_counts = np.flip(new_counts)
        new_edges = np.flip(new_edges)
        
    return np.array(new_counts), np.array(new_edges)


def cat_sx_dx(sx: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """
    Concatenate arrays of merged bins on the left and right of the mode.

    Parameters
    ----------
    sx : np.ndarray
        Array of merged bins on the left of the mode.
    dx : np.ndarray
        Array of merged bins on the right of the mode.

    Raises
    ------
    Exception
        If both arrays are empty, there must have been something wrong with
        merging or sampling.

    Returns
    -------
    np.ndarray
        Array of merged bins along the whole histogram.

    """
    if sx.shape[0] == 0:
        return dx
    elif dx.shape[0] == 0:
        return sx
    elif sx.shape[0] != 0 and dx.shape[0] != 0:
        return np.concatenate([sx, dx])
    else:
        raise Exception('There must be something wrong with sampling.')


def re_bin(sample: np.ndarray, n_bins: int = 10, th: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-bin a sample based on the mode of its distribution.

    Parameters
    ----------
    sample : np.ndarray
        The original sample data.
    n_bins : int, optional
        Number of bins for the original histogram. Default is 10.
    th : int, optional
        Minimum number of samples per bin. The default is 5.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Merged counts and edges of the re-binned histogram.

    """
    
    # original histogram of the sample
    counts, edges = np.histogram(sample, bins=n_bins)
        
    # find a more convenient representation of bins
    edges = unravel_edges(edges)
    
    # find the index of the mode
    id_max = list(counts).index(counts.max())
    
    # split data wrt the mode of the distribution
    sx_edges = edges[None:id_max,:]
    dx_edges = edges[id_max:None,:]
    sx_counts = counts[None:id_max]
    dx_counts = counts[id_max:None]
    
    # on the left
    sx_new_counts, sx_new_edges = merge_bins(sx_counts, sx_edges, th = th)
    # on the right -- watch out: flip the array
    dx_new_counts, dx_new_edges = merge_bins(dx_counts, dx_edges, th = th, flip = True)
    
    # join new counts and bins
    new_counts = cat_sx_dx(sx_new_counts, dx_new_counts)
    new_edges = cat_sx_dx(sx_new_edges, dx_new_edges)
    
    # in order to plot take the edges of the distribution
    new_edges_plot = np.array([new_edges[0,0]] + list(new_edges[:,1]))
    
    print(f"\nMinimum bin size: {th:d}\n")
    print(f"Original numerosity per bin\n{counts}\n")
    print(f"Merged numerosity per bin\n{new_counts}\n")

    return new_counts, new_edges, new_edges_plot