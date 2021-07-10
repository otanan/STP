#!/usr/bin/env python3
"""Calculates a typical set for provided systems, incorporating various methods of path space sampling to construct a representative set in a reasonable amount of time.

**Author: Jonathan Delgado**

"""

######################## Imports ########################


import numpy as np

import stp
import stp.info as info
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


class TypicalSet:
    """ Typical set object. Holds path probabilities, typical paths, atypical 
        paths, atypical path probabilities and more.
    
        This object will use a provided (often sampled) path space to partition the space into a collection of typical and atypical paths depending on the dynamics provided. Will also track other quantities of interest such as the upper and lower bounds on the path probabilities required for the paths to be considered typical.
        
        Attributes:
            epsilon: the width of the neighborhood used for paths to be considered typical.

            num_paths: the number of paths considered (typical + atypical).

            path_length: the length of the paths considered.

            paths: a numpy matrix holding paths as rows and the nth column will correspond to the nth state observed for each path.

            probabilities: a matrix where the (i,j)th element is the probability of observing the first j states of the ith path.

            path_typicalities: a matrix where the (i,j)th element is a boolean determining whether the ith path is typical after j steps.

            upper_bound: a list of upper bounds versus symbols observed corresponding to the upper bound for paths probabilities for the path to be considered typical.

            lower_bound: the lower bound analogous to the upper bound.
    """
    def __init__(self, R, p, paths, epsilon=1):
        """ Samples a path space and generates a typical set on this space.

            Args:
                R (np.ndarray/function): the transition matrix, time-dependent if provided as a function

                p (np.ndarray): the initial marginal distribution

                paths (np.ndarray): the portion of the path space to use.
        
            Kwargs:
                epsilon (float): the epsilon neighborhood to consider paths to be typical within
        
        """
        self.epsilon = epsilon
        self.paths = paths

        #------------- Data preparation -------------#

        # Convert the transition matrix to add time-dependence as a constant 
            # matrix if a constant matrix was provided
        if not callable(R):
            # Not being saved as an attribute since this is not easily
                # recoverable by being saved to a file.
            # Emphasize saving properties that can be saved/loaded.
            oldR = R
            R = lambda n : oldR

        # The number of states
        n = len(p)

        self.num_paths, self.path_length = paths.shape

        self.probabilities = np.zeros(paths.shape)
        # Initialize the marginal distribution data
        for x, path in enumerate(self.paths):
            # Just equal to the initial marginal
            self.probabilities[x, 0] = p[path[0]]

        # Used for the bounds
        entropy_rate = info.entropy_rate(R)
        entropy_rate_vs_time = np.array([
            entropy_rate(i)
            for i in range(self.path_length)
        ])

        #------------- Data gathering -------------#

        bar = gui.ProgressBar(self.path_length * self.num_paths, width=300, title='Gathering data...')
        
        ### Quantities versus time ###
        for current_path_length in range(2, self.path_length + 1):
            # The data index
            i = current_path_length - 1
            # Since the marginals are zero-indexed as are the paths
            step_index = current_path_length - 2

            currentR = R(current_path_length - 1)
            # Propagate the marginal one step and save it separately
                # for quantities like the temporal coarse graining term
            pstep = stp.step(currentR, p)

            ### Path probability calculations ###
            for x, path in enumerate(paths):

                current_state = path[step_index]
                jump_state = path[step_index + 1]

                # Forward calculations
                # Recursive calculation to save time
                last_joint = self.probabilities[x, i - 1]
                jump_prob = currentR[jump_state, current_state]
                self.probabilities[x, i] = last_joint * jump_prob

                bar.update()

            # Finished data gathering for this iteration, propagate marginal
                # forward in time
            p = pstep

        bar.finish()

        #------------- Typical set bounds and generation -------------#

        ns = np.arange(1, self.path_length + 1)

        self.lower_bound = np.exp(-ns * (entropy_rate_vs_time + self.epsilon))
        self.upper_bound = np.exp(-ns * (entropy_rate_vs_time - self.epsilon))

        self.path_typicalities = []

        # Identify the paths that are typical and atypical
        for path_probs in self.probabilities:

            # Array of bools determining when a path is typical or atypical
            self.path_typicalities.append(
                (self.lower_bound < path_probs) \
                & (path_probs < self.upper_bound)
            )


#------------- Entry code -------------#


def main():
    print('typical_set.py')
    
    

if __name__ == '__main__':
    main()
