#!/usr/bin/env python3
"""Entropy and information theory related calculations.

**Author: Jonathan Delgado**

"""

######################## Imports ########################


import numpy as np

import stp


######################## Helper functions ########################


def _eps_filter(x):
    """ Checks if the value is within machine-epsilon of zero and maps it to 
        zero if it is the case. Useful for removing negative values in entropies that should otherwise be zero.
        
        Args:
            x (float): value to be checked.
    
        Returns:
            (float): x if the value is not within machine epsilon of zero, 0 otherwise.
    
    """
    return x if not np.isclose(x, 0, atol=9*10E-15) else 0
    

######################## Entropy calculations ########################


def entropy(p):
    """ Calculates the Shannon entropy for a marginal distribution.

        Args:
            p (np.ndarray): the marginal distribution.

        Returns:
            (float): the entropy of p

    """
    # Since zeros do not contribute to the Shannon entropy by definition, we 
        # ignore them to avoid any errors/warnings.
    p = p[p != 0]

    H = -np.dot(p, np.log(p))
    # Filter against machine epsilon
    return _eps_filter(H)


def delta_entropy(R, p):
    """ Calculates the discrete time change in entropy using the entropy of p 
        evolved with R, minus the entropy of p.
    
        Args:
            R (np.ndarray): the transition matrix.

            p (np.ndarray): the marginal distribution.
    
        Returns:
            (float): the change in entropy
    
    """
    return entropy(step(R, p)) - entropy(p)


def relative_entropy(p, q):
    """ Calculates the Kullback-Leibler divergence, which is nonnegative and 
        vanishes if and only if the distributions coincide.
    
        Args:
            p, q (np.ndarray): the probability distributions.
    
        Returns:
            (float): the relative entropy.
    
    """
    if p.shape[0] != q.shape[0]:
        print('Dimensions of vectors are not equal. Cannot find relative entropy.')
        sys.exit()

    # Any values where p is zero are defined to be zero and hence do not
        # contribute to the relative entropy
    # By masking q as well we automatically skip the values that were supposed
        # to vanish with p avoiding any misalignment issues
    # Note that by masking q only where p is zero doesn't remove
        # any mismatching meaning it will still be infinite (as it should be)
        # in the case where q has a zero that p does not.
    p_filtered = p[p != 0]
    log_ratio = np.log(p_filtered / q[p != 0])

    return np.dot(p_filtered, log_ratio)


def entropy_production(matrix, p, discrete=True):
    """ Calculates the entropy production for either discrete or continuous 
        time.
    
        Args:
            matrix (np.ndarray): the stochastic matrix, either a discrete time transition matrix or a continuous time rate matrix.

            p (np.ndarray): the marginal distribution

        Kwargs:
            discrete (bool): True if we are calculating the discrete time entropy production (nats), False if we are calculating it in continuous time (nats/time).
    
        Returns:
            (float/np.inf): the entropy production
    
    """
    log_product = matrix * np.log( matrix / matrix.T )
    # The entropy term only exists in the case of discrete time
        # it vanishes when we calculate the continuous time EP,
        # by multiplying by the boolean we include it only when
        # necessary
    EP = np.dot(log_product.sum(axis=0), p) - (entropy(p) * discrete) \
        - np.dot(stp.step(matrix, p), np.log(p))
    return EP


def entropy_flow(R, p):
    """ Calculates the discrete time entropy flow. This has not been    
        generalized to handle the continuous time entropy flow yet.
    
        Args:
            R (np.ndarray): the discrete time transition matrix
            
            p (np.ndarray): the marginal distribution
    
        Returns:
            (float): the entropy flow
    
    """
    # Vectorized calculation
    log_product = R * np.log( R / R.T )
    p_step = step(R, p)
    EF = -np.dot(log_product.sum(axis=0), p) + entropy(p_step) \
        + np.dot(p_step, np.log(p))
    return EF


######################## Entropy rates ########################


def entropy_rate(R):
    """ Calculates the asymptotic entropy rate for the provided transition 
        matrix. If the matrix is time-inhomogeneous then we return a function that generates the entropy_rate as a function of n by calculating the systems limiting distribution for each n.
        
        Args:
            R (np.ndarray/function): the transition matrix.
    
        Returns:
            (float/function): the entropy velocity.
    
    """
    if callable(R):
        return lambda n : entropy_rate(R(n))

    pst = stp.get_stationary_distribution(R, discrete=True)
    RProduct = (R * np.log(R)).sum(axis=0)

    return -np.dot(pst, RProduct)


#------------- Entry code -------------#

def main():
    print('info.py')
    

if __name__ == '__main__':
    main()
