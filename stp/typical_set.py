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


######################## Information Space Object ########################

class InfoSpace:
    """ Information space. Holds collections of paths that traverse states in a 
        state space as a matrix, and the probability of each of those paths. 
        
        Provides functionality on this path space such as providing path entropies.
        
        Attributes:
            paths: the matrix of paths.

            probabilities: a list of probabilities each path.

            num_paths: the number of paths considered.

            path_length: the length of the paths considered.

            probabilities: a matrix where the (i,j)th element is the probability of observing the first j states of the ith path.

            entropies: a list of path entropies for each path
    """

    def __init__(self, paths, p_matrix):
        """ Initializes the InfoSpace object.
            
            Args:
                paths (np.ndarray): a matrix of paths where the (i,j)th element corresponds to the jth symbol of the ith path.

                p_matrix (np.ndarray): a matrix of probabilities where the (i,j)th element corresponds to the probability of observing the ith path for the first j+1 (zero-indexing) symbols.
        
        """
        self._paths = np.array(paths)
        # Matrix of probabilities corresponding to the probability for the path
            # at each moment.
        self._p_matrix = np.array(p_matrix)
        
        if self._p_matrix.size != 0:
            # The typical set is not empty
            self._probabilities = self._p_matrix[:, -1]


    #------------- Properties -------------#


    @property
    def paths(self):
        return self._paths


    @property
    def num_paths(self):
        return self.paths.shape[0]


    @property
    def path_length(self):
        return self.paths.shape[1]
    

    @property
    def probabilities(self):
        return self._probabilities


    @property
    def entropies(self):
        """ Returns a list of path entropies for each corresponding path 
            probability.

        """
        try:
            return self._entropies
        except AttributeError:
            # It's never been calculated before
            self._entropies = -np.log(self.probabilities)

        return self._entropies


    #------------- Static methods -------------#


    @staticmethod
    def shorten(infospace, path_length):
        """ Takes an Information Space and shortens it. Since unique paths of 
            length n, may be degenerate when truncated to paths of length m < n, we need to check for degeneracies and filter them out in both paths and probabilities.
            
            Args:
                infospace (InfoSpace): the information space to shorten.

                path_length (int): the path length the information space should be shortened to.        
        
            Returns:
                (InfoSpace): the shortened InfoSpace.
        
        """
        # Truncate the path matrix
        paths = infospace.paths[:, :path_length]
        # Return index will provide the path indices of the non-degenerate paths
        _, indices = np.unique(paths, axis=0, return_index=True)
        # Sort the indices
        indices = sorted(indices)
        # Filter out the paths. Not taken from np.unique to ensure the correct
            # ordering.
        paths = paths[indices, :]
        # Truncate the probability matrix
        p_matrix = infospace._p_matrix[:, :path_length]
        # Filter the probabilities matrix
        p_matrix = p_matrix[indices, :]

        return InfoSpace(paths, _p_matrix)


######################## Main body ########################


class PartitionedInfoSpace(InfoSpace):
    """ Partitioned Information Space. Constructs a typical set on an 
        information space to partition it into a typical information space and an atypical one. 

        Holds path probabilities, typical paths, atypical paths, atypical path probabilities and more. This object will use a provided (often sampled) path space to partition the space into a collection of typical and atypical paths depending on the dynamics provided. Will also track other quantities of interest such as the upper and lower bounds on the path probabilities required for the paths to be considered typical.
    
        Attributes:
            paths: the matrix of paths.

            probabilities: a list of probabilities each path.

            num_paths: the number of paths considered.

            path_length: the length of the paths considered.

            probabilities: a matrix where the (i,j)th element is the probability of observing the first j states of the ith path.

            entropies: a list of path entropies for each path.

            entropy_rates: a list of the entropy rates for each various path length. This will be the center of the epsilon-neighborhood for path entropies to qualify paths as typical for.

            epsilon: the widths of the neighborhood used for paths to be considered typical for each path length.

            upper/lower: the upper/lower bounds on path entropies for a path to be considered typical for each path length.

            typicalities: a matrix where the (i,j)th element is a boolean determining whether the ith path is typical after j+1 steps.

            ts: the typical set.

            ats: the atypical set.

    """

    def __init__(self, typical_space, atypical_space, entropy_rates, epsilon, typicalities=None):
        """ Generates the PartitionedInfoSpace.
            
            Args:
                typical_space (InfoSpace): the typical set on this space.

                atypical_space (InfoSpace): the atypical set on this space.

                entropy_rates (np.ndarray): a list of the entropy rates for each various path length. This will be the center of the epsilon-neighborhood for path entropies to qualify paths as typical for.

                epsilon (np.ndarray): the widths of the neighborhood used for paths to be considered typical for each path length.

            Kwarrg:
                typicalities (np.ndarray/None): a matrix where the (i,j)th element is a boolean determining whether the ith path is typical after j+1 steps. Calculated from other quantities if None is provided.
        
        """
        # Combine the path data
        if (typical_space.paths.size != 0) and (atypical_space.paths.size != 0):
            # Both are nonempty
            self._paths = np.vstack((typical_space.paths, atypical_space.paths))
            self._p_matrix = np.vstack((typical_space._p_matrix, atypical_space._p_matrix))
        elif typical_space.paths.size == 0:
            # Only the typical_space is empty
            self._paths = atypical_space.paths
            self._p_matrix = atypical_space._p_matrix
        else:
            # Only the atypical_space is empty
            self._paths = typical_space.paths
            self._p_matrix = typical_space._p_matrix

        self._probabilities = self._p_matrix[:, -1]

        self._entropy_rates = entropy_rates

        if isinstance(epsilon, list):
            epsilon = np.array(epsilon)
        if not isinstance(epsilon, np.ndarray):
            # We were only provided a float
            epsilon = np.full(self.path_length, epsilon)
        self._epsilon = epsilon

        self._typicalities = typicalities

        self._ts = typical_space
        self._ats = atypical_space


    #------------- Properties -------------#


    @property
    def entropy_rates(self):
        return self._entropy_rates
    

    @property
    def epsilon(self):
        return self._epsilon


    @property
    def upper(self):
        try:
            return self._upper
        except AttributeError:
            # It's never been calculated before
            ns = np.arange(1, self.path_length + 1)

            self._upper = -ns * (self.entropy_rates - self.epsilon)
            # Lower is calculated similarly, so do it now
            self._lower = -ns * (self.entropy_rates + self.epsilon)

        return self._upper


    @property
    def lower(self):
        try:
            return self._lower
        except AttributeError:
            # It's never been calculated before.
            # self.upper calculates the lower bound too.
            # Trigger the calculation.
            self.upper

        return self._lower


    @property
    def typicalities(self):
        """ Returns the matrix of typicalities. """
        if self._typicalities == None:
            # It's never been calculated before
            typicalities = []

            for path_entropy in -np.log(self._p_matrix):
                # Check if and when the path is typical and append it
                typicalities.append(
                    (self.lower < path_entropy) & (path_entropy < self.upper)
                )

            self._typicalities = np.array(typicalities)

        return self._typicalities


    @property
    def ats(self):
        return self._ats


    @property
    def ts(self):
        return self._ts
    
    
    #------------- Static methods -------------#


    @staticmethod
    def shorten(pinfospace, path_length):
        """ Takes a PartitionedInformationSpace and shortens it. Since unique 
            paths of length n, may be degenerate when truncated to paths of length m < n, we need to check for degeneracies and filter them out in both paths and probabilities.
            
            Args:
                pinfospace (PartitionedInfoSpace): the partitioned information space to shorten.

                path_length (int): the path length the information space should be shortened to.        
        
            Returns:
                (PartitionedInfoSpace): the shortened PartitionedInfoSpace.
        
        """
        # Truncate the path matrix
        paths = pinfospace.paths[:, :path_length]
        # Return index will provide the path indices of the non-degenerate paths
        _, indices = np.unique(paths, axis=0, return_index=True)
        # Sort the indices
        indices = sorted(indices)
        # Filter out the paths. Not taken from np.unique to ensure the correct
            # ordering.
        paths = paths[indices, :]

        # Truncate the probability matrix
        p_matrix = pinfospace._p_matrix[:, :path_length]
        # Filter the probabilities matrix
        p_matrix = p_matrix[indices, :]
        
        # Truncate the entropy_rates
        entropy_rates = pinfospace.entropy_rates[:path_length]

        # Truncate the epsilon
        epsilon = pinfospace.epsilon[:path_length]

        # Truncate the typicalities matrix
        # Necessary to re-partition the space.
        typicalities = pinfospace.typicalities[:, :path_length]
        # Filter out the typicalities matrix
        typicalities = typicalities[indices, :]

        ### Partitioning ###
        ts_paths, ts_p_matrix = [], []
        ats_paths, ats_p_matrix = [], []

        for path_index, is_typical in enumerate(typicalities[:, -1]):
            path = path[path_index]
            probs = p_matrix[path_index]

            if is_typical:
                ts_paths.append(path)
                ts_p_matrix.append(probs)
            else:
                ats_paths.append(path)
                ats_p_matrix.append(probs)

        # The partitioned spaces
        ts = InfoSpace(ts_paths, ts_p_matrix)
        ats = InfoSpace(ats_paths, ats_p_matrix)

        return PartitionedInfoSpace(ts, ats, entropy_rates, epsilon, typicalities)


    @staticmethod
    def partition_space(R, p, paths, epsilon=1):
        """ Partitions a path space using the dynamics provided.

            Args:
                R (np.ndarray/function): the transition matrix, time-dependent if provided as a function.

                p (np.ndarray): the initial marginal distribution.

                paths (np.ndarray): the portion of the path space to use.
        
            Kwargs:
                epsilon (float/np.ndarray): the radius/radii of the epsilon neighborhood to consider paths to be typical within.
        
            Returns:
                (2-tuple): the PartitionedInfoSpace and a list of the marginal versus observation step.

        """

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

        num_paths, path_length = paths.shape

        p_matrix = np.zeros(paths.shape)
        # Initialize the marginal distribution data
        for x, path in enumerate(paths):
            # Just equal to the initial marginal
            p_matrix[x, 0] = p[path[0]]

        # Used for the bounds
        entropy_rate = info.entropy_rate(R)
        entropy_rates = np.array([
            entropy_rate(i)
            for i in range(path_length)
        ])

        # The marginal versus time
        p_vs_time = [p]

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
            pstep = stp.step(currentR, p)

            ### Path probability calculations ###
            for x, path in enumerate(paths):

                current_state = path[step_index]
                jump_state = path[step_index + 1]

                # Forward calculations
                # Recursive calculation to save time
                last_joint = p_matrix[x, i - 1]
                jump_prob = currentR[jump_state, current_state]
                p_matrix[x, i] = last_joint * jump_prob

                bar.update()

            # Finished data gathering for this iteration, propagate marginal
                # forward in time
            p_vs_time.append(pstep)
            p = pstep

        bar.finish()

        ### Partitioning ###

        ns = np.arange(1, path_length + 1)

        lower = -ns * (entropy_rates + epsilon)
        upper = -ns * (entropy_rates - epsilon)

        typicalities = []

        ts_paths, ts_p_matrix = [], []
        ats_paths, ats_p_matrix = [], []

        # Identify the paths that are typical and atypical
        for path_index, path_entropy in enumerate(-np.log(p_matrix)):

            typicalities.append(
                (lower < path_entropy) & (path_entropy < upper)
            )

            probs = p_matrix[path_index, :]

            # Whether this path is typical
            is_typical = typicalities[-1][-1]

            if is_typical:
                ts_paths.append(path)
                ts_p_matrix.append(probs)
            else:
                ats_paths.append(path)
                ats_p_matrix.append(probs)

        # The partitioned spaces
        ts = InfoSpace(ts_paths, ts_p_matrix)
        ats = InfoSpace(ats_paths, ats_p_matrix)

        pinfospace = PartitionedInfoSpace(ts, ats, entropy_rates, epsilon, typicalities)

        # Set pre-calculated properties
        pinfospace._upper = upper
        pinfospace._lower = lower

        return pinfospace, p_vs_time


#------------- Entry code -------------#


def main():
    print('typical_set.py')

    ### Testing ###
    p = stp.rand_p(3)
    R = stp.self_assembly_transition_matrix()
    paths = stp.complete_path_space(3, 4)
    pinfospace, _ = PartitionedInfoSpace.partition_space(R, p, paths)
    print(pinfospace.ats.num_paths)
    
    

if __name__ == '__main__':
    main()
