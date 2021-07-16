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
    def shorten(infospace, path_length, return_index=False):
        """ Takes an Information Space and shortens it. Since unique paths of 
            length n, may be degenerate when truncated to paths of length m < n, we need to check for degeneracies and filter them out in both paths and probabilities.
            
            Args:
                infospace (InfoSpace): the information space to shorten.

                path_length (int): the path length the information space should be shortened to.

            Kwargs:
                return_index (bool): returns the indices of the non-degenerate paths for the given path length using the original matrix. Useful for filtering other quantities of interest that may not be attached to this object.      
        
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

        infospace = InfoSpace(paths, _p_matrix)
        return infospace if not return_index else infospace, indices


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

            upper/lower: the upper/lower bounds as measured in nats. This means that a path is typical if and only if its path entropy rate is within these bounds.

            typicalities: a matrix where the (i,j)th element is a boolean determining whether the ith path is typical after j+1 steps.

            ts: the typical set.

            ats: the atypical set.

    """

    def __init__(self, typical_space, atypical_space, entropy_rates, epsilon):
        """ Generates the PartitionedInfoSpace.
            
            Args:
                typical_space (InfoSpace): the typical set on this space.

                atypical_space (InfoSpace): the atypical set on this space.

                entropy_rates (np.ndarray): a list of the entropy rates for each various path length. This will be the center of the epsilon-neighborhood for path entropies to qualify paths as typical for.

                epsilon (np.ndarray): the widths of the neighborhood used for paths to be considered typical for each path length.
        
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
            
            self._upper = self.entropy_rates + self.epsilon
            # Lower is calculated similarly, so do it now
            self._lower = self.entropy_rates - self.epsilon

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
        try:
            return self._typicalities
        except AttributeError:
            # It's never been calculated before
            typicalities = []
            ns = np.arange(1, self.path_length + 1)

            for path_entropy in -np.log(self._p_matrix):
                path_entropy_rate = path_entropy / ns
                # Check if and when the path is typical and append it
                typicalities.append(
                    (self.lower < path_entropy_rate) & (path_entropy_rate < self.upper)
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
    def shorten(pinfospace, path_length, return_index=False):
        """ Takes a PartitionedInformationSpace and shortens it. Since unique 
            paths of length n, may be degenerate when truncated to paths of length m < n, we need to check for degeneracies and filter them out in both paths and probabilities.
            
            Args:
                pinfospace (PartitionedInfoSpace): the partitioned information space to shorten.

                path_length (int): the path length the information space should be shortened to.    

            Kwargs:
                return_index (bool): returns the indices of the non-degenerate paths for the given path length using the original matrix. Useful for filtering other quantities of interest that may not be attached to this object.    
        
            Returns:
                (PartitionedInfoSpace): the shortened PartitionedInfoSpace.
        
        """
        # Call parent method
        # Paths and p_matrix will be handled here along with any other 
            # properties shared with parent. Sorted indices of non-degenerate 
            # paths will be calculated here too.
        pinfospace, indices = InfoSpace.shorten(pinfospace, path_length, return_index=True)
        
        # Finish the rest of this object's specific properties

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

        pinfospace = PartitionedInfoSpace(ts, ats, entropy_rates, epsilon, typicalities)
        return pinfospace if not return_index else pinfospace, indices


    @staticmethod
    def partition_space(R, p, paths, epsilon=0.5, return_p=False):
        """ Partitions a path space using the dynamics provided.

            Args:
                R (np.ndarray/function): the transition matrix, time-dependent if provided as a function.

                p (np.ndarray): the initial marginal distribution.

                paths (np.ndarray): the portion of the path space to use.
        
            Kwargs:
                epsilon (float/np.ndarray): the radius/radii of the epsilon neighborhood to consider paths to be typical within.

                return_p (bool): False, return only the PartitionedInfoSpace, True returns both the PartitionedInfoSpace and a list of the marginal vs time.
        
            Returns:
                (ParitionedInfoSpace/2-tuple): the PartitionedInfoSpace (PIS) or the PIS and a list of the marginal versus observation step if return_p is True.

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

        num_paths, path_length = paths.shape

        p_matrix = np.zeros(paths.shape)
        # Initialize the marginal distribution data
        for x, path in enumerate(paths):
            # Just equal to the initial marginal
            p_matrix[x, 0] = p[path[0]]
        
        # Used for the bounds
        entropy_rates = np.array([
            info.entropy_rate(R(i))
            for i in range(path_length)
        ])

        # The marginal versus time
        if return_p: p_vs_time = [p]

        #------------- Data gathering -------------#

        # bar = gui.ProgressBar(path_length * num_paths, width=300, title='Gathering data...')

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

            # If updated in each iteration, slows down the simulation 
                # drastically
            # bar.update(amount=num_paths)

            if return_p: p_vs_time.append(pstep)
            # Finished data gathering for this iteration, propagate marginal
                # forward in time
            p = pstep

        # bar.finish()

        ### Partitioning ###

        ns = np.arange(1, path_length + 1)

        upper = entropy_rates + epsilon
        lower = entropy_rates - epsilon

        ts_paths, ts_p_matrix = [], []
        ats_paths, ats_p_matrix = [], []

        # Identify the paths that are typical and atypical
        for path_index, path_entropy in enumerate(-np.log(p_matrix[:, -1])):

            path_entropy_rate = path_entropy / path_length

            # Can't create typicality matrix since partitioning it will
                # break the ordering
            # Determines whether this path is ultimately typical
            is_typical = (lower[-1] < path_entropy_rate) and (path_entropy_rate < upper[-1])

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

        pinfospace = PartitionedInfoSpace(ts, ats, entropy_rates, epsilon)

        # Set pre-calculated properties
        pinfospace._upper = upper
        pinfospace._lower = lower

        return (pinfospace, p_vs_time) if return_p else pinfospace


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
