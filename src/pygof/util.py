import numpy as np
from matplotlib import pyplot as plt


def random_variable(distribution, **args) -> None:
    """
    Generate a random variable from a specified distribution.

    Parameters
    ----------
    distribution : callable
        The probability distribution function to sample from. (scipy.dist)
    **args : keyword arguments
        Parameters specific to the chosen distribution function.

    Returns
    -------
    variable
        A random variable sampled from the specified distribution.

    """
    return distribution(**args)


def inspect_sample(sample: np.ndarray = None, n_bins: int = 50,
                   rv: callable = None, res: int = 1000, density: bool = False,
                   new_counts: np.ndarray = None, new_edges: np.ndarray = None) -> None:
    """
    Visualize the original and merged histograms of a sample, along with a probability density function (PDF).

    Parameters
    ----------
    sample : np.ndarray, optional
        The original sample data. Default is None.
    n_bins : int, optional
        Number of bins for the original histogram. Default is 50.
    rv : callable, optional
        Probability density function (PDF) of the distribution. Default is None.
    res : int, optional
        Number of points for generating the PDF curve. Default is 1000.
    density : bool, optional
        If True, the histogram and PDF are normalized to form a probability density. Default is False.
    new_counts : np.ndarray, optional
        Array of merged bin counts. Default is None.
    new_edges : np.ndarray, optional
        Array of merged bin edges. Default is None.

    Returns
    -------
    None
   
    """
 
    if sample is not None:
        plt.figure(dpi=300)
        plt.hist(sample, bins=n_bins, histtype='step', color='red', density=density, label='Original')
        
        if rv is not None: 
            x = np.linspace(sample.min(), sample.max(), res)
            plt.plot(x, rv.pdf(x), 'k', label='PDF')
            
    if new_counts is not None and new_edges is not None:
        plt.stairs(new_counts, new_edges, fill=True, color = 'blue', label='Merged')
          
    if density:
        plt.title('Relative Frequency')
    else:
        plt.title('Absolute Frequency')
    
    plt.legend()
    plt.show()
