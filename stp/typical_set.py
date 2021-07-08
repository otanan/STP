#!/usr/bin/env python3
"""Calculates a typical set for provided systems, incorporating various methods of path space sampling to construct a representative set in a reasonable amount of time.

**Author: Jonathan Delgado**

"""

######################## Imports ########################


import numpy as np

import stp
import stp.info
# For ProgressBar
import stp.tools.gui as gui


######################## Helper functions ########################


def probability_captured(typical_set, sampled_set):
    """ Calculates the amount of probability captured by the typical set
        and performs any relevant analysis.
        
        Args:
            typical_set (list): the typical set.

            sampled_set (list): the sampled path space.
    
        Returns:
            (float): the probability captured by the typical set.
    
    """
    TS_probability = sum( [ entry[1][-1] for entry in typical_set ] )
    print( f'Probability captured by the sampled typical set: {TS_probability}.' )
    
    sampled_probability = sum( [ entry[1][-1] for entry in sampled_set ] )
    print( f'Total probability sampled: {sampled_probability}.' )
    
    print( f'Percentage of sampled probability that is typical: {TS_probability / sampled_probability * 100:.1f}%.' )

    return TS_probability


def verify_TS(typical_set, R, epsilon):
    """ Tests Theorem 3.1.2 of Cover & Thomas.
        
        Args:
            typical_set (list): a list of 2-tuples where the first component is the path, and the second component is the path probability as a function of n.All paths in this list are typical.

            R (np.ndarray): the transition matrix used to calculate the limiting distribution for the system and hence its entropy rate. If it is time-inhomogeneous, the typical set will be used to pull out the relevant transition matrix.

            epsilon (float): the epsilon being used
    
        Returns:
            (None): none
    
    """
    num_typical = len(typical_set)
    # All path lengths are the same so just check how long the first path is.
    if num_typical > 0:
        n = len(typical_set[0][0])
    else:
        # n can just be assigned since the relevant theorem will be False.
        n = 1000

    if callable(R):
        # The matrix is time-inhomogeneous.
        # We just need the transition matrix corresponding to the final entropy
            # rate
        R = R(n)

    # Theorem 3.1.2:
    # 2. Pr A(n) > 1 − epsilon for n sufficiently large.
    # 3. |A(n)| ≤ 2^{n( H(X)+epsilon )}
    # 4. |A(n)| ≥ (1 − epsilon)2^{n( H(X)−epsilon )} for n sufficiently large.

    # Get the probability captured by the typical set at the final observation
    TS_probability = sum( [ entry[1][-1] for entry in typical_set ] )

    entropy_rate = info.entropy_rate(R)

    ### Printing of test results. ###

    print('NOTE: Only Theorem 3.1.2.3 is REQUIRED to be true for these simulations. It is not necessarily incorrect for 3.1.2.2 or 3.1.2.4 to be False.')
    print()
    print(f'Probability of observing a typical path: {TS_probability}.')
    print(f'1 - epsilon: {1 - epsilon}.')
    print(f'Theorem 3.1.2.2 satisfied: {TS_probability > 1 - epsilon}.')
    print()
    print(f'Number of typical paths: {num_typical}.')
    print(f'Entropy rate: {entropy_rate}.')
    upper_bound = 2**(n * (entropy_rate + epsilon))
    print(f'Theorem 3.1.2.3 satisfied: {upper_bound > num_typical}.')
    print()
    lower_bound = (1-epsilon) * 2**(n * (entropy_rate - epsilon))
    print(f'Theorem 3.1.2.4 satisfied: {num_typical > lower_bound}.')


######################## Main body ########################


def typical_set(R, p, path_space, time_step=1, epsilon=1):
    """ Samples a path space and generates a typical set on this sampled path 
        space.

        Args:
            R (np.ndarray/function): the transition matrix, time-dependent if provided as a function

            p (np.ndarray): the initial marginal distribution

            path_space (np.ndarray): the portion of the path space to use.
    
        Kwargs:
            time_step (float): the time between observations

            epsilon (float): the epsilon neighborhood to consider paths to be typical within
    
        Returns:
            (2-tuple): the first component being the typical set as a list of 2-tuples, each tuple being the (path, path probability vs n), the second component of the returned tuple is the total path space as a list of 3-tuples, each tuple being (path, path probability vs n, True if typical False if a typical as a function of n). In this case a path is added to the typical set if its final path entropy rate is within the typical set bounds. While a path may be atypical by the end of the observation, it could have been typical before, and this information is encoded in the last component as a list of bools.
    
    """
     
    #------------- Data preparation -------------#

    # Convert the transition matrix to add time-dependence as a constant matrix
        # if a np.ndarray was provided
    if isinstance(R, np.ndarray):
        oldR = R
        R = lambda n : oldR

    # The number of states
    n = len(p)

    final_time = time_step * (path_length - 1)
    num_paths, path_length = path_space.shape

    # Save the initial distribution
    pinit = p

    # List of 2-tuples representing the typical set
        # (path_probability, forward path)
    typical_set = []
    # List of 3-tuples representing the sampled path space
        # (path_probability, forward path, is_typical)
    sampled_set = []

    path_probabilities = np.zeros(paths.shape)

    # Used for the bounds
    entropy_rate = info.entropy_rate(R)
    entropy_rate_vs_time = [
        entropy_rate(i)
        for i in range(path_length)
    ]

    #------------- Data gathering -------------#

    bar = gui.ProgressBar(path_length * num_paths, width=300, title='Gathering data...')
    
    ### Quantities versus time ###
    for current_path_length in range(2, path_length + 1):
        # The data index
        i = current_path_length - 1
        # Since the marginals are zero-indexed as are the paths
        step_index = current_path_length - 2

        currentR = R(current_path_length - 1)
        # Propagate the marginal one step and save it separately
            # for quantities like the temporal coarse graining term
        pstep = step(currentR, p)

        ### Path probability calculations ###
        for x, path in enumerate(paths):

            current_state = path[step_index]
            jump_state = path[step_index + 1]

            # Forward calculations
            # Recursive calculation to save time
            last_joint = path_probabilities[x, i - 1]
            jump_prob = currentR[jump_state, current_state]
            path_probabilities[x, i] = last_joint * jump_prob

            bar.update()


        # Finished data gathering for this iteration, propagate marginal
            # forward in time
        p = pstep


    bar.finish()


    #------------- Typical set bounds and generation -------------#

    TS_upper = entropy_rate_vs_time + epsilon
    TS_lower = entropy_rate_vs_time - epsilon

    path_entropy_rates = -np.log(path_probabilities) / ns

    # Identify the paths that are typical and atypical
    # Populate the sets
    for path_index, path_entropy_rate in enumerate(path_entropy_rates):

        # Information for typical set and sampled_set
        path_info = [
            paths[path_index],
            # The list of the probability this path observed with each
                # added state
            path_probabilities[path_index],
            # List of bools where the True means the path is typical at this
                # step, False if it is atypical
            # If the last element is True then this tuple is added to the
                # typical set
            (TS_lower < path_entropy_rate) & (path_entropy_rate < TS_upper)
        ]

        # The path is considered typical at the end of the observation
        # Append the 3-tuple to the typical set
        if path_info[-1][-1]:
            typical_set.append( tuple(path_info[:-1]) )

        sampled_set.append( tuple(path_info) )


    #------------- Analysis & Printing -------------#

    percent_typical = f'{len(typical_set) / num_paths * 100:.1f}'
    print(f'{percent_typical}% of tested paths are typical.')


    return typical_set, sampled_set


#------------- Entry code -------------#


def main():
    print('typical_set.py')
    

if __name__ == '__main__':
    main()
