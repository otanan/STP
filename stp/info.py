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


######################## Information Space Objects ########################


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

            total_probability: the sum of the probabilities of each path.

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
            # The information space is not empty
            self._probabilities = self._p_matrix[:, -1]
        else:
            # There is zero probability here.
            self._probabilities = 0


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


    @property
    def total_probability(self):
        try:
            return self.probabilities.sum()
        except AttributeError:
            # Space is empty
            return 0
    

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
        if path_length < 1:
            raise ValueError(f'Invalid path length: {path_length}. Path length must be an integer greater than 0.')
        elif path_length > infospace.path_length:
            raise ValueError(f'Cannot shorten an InformationSpace from length: {infospace.path_length} -> {path_length}.')

        if infospace.paths.size == 0:
            # This is an empty information space
            return infospace if not return_index else (infospace, [])

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

        infospace = InfoSpace(paths, p_matrix)
        return infospace if not return_index else infospace, indices


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

    def __init__(self, entropy_rates, epsilon, paths=None, p_matrix=None, typical_space=None, atypical_space=None):
        """ Generates the PartitionedInfoSpace.
            
            Args:
                entropy_rates (np.ndarray): a list of the entropy rates for each various path length. This will be the center of the epsilon-neighborhood for path entropies to qualify paths as typical for.

                epsilon (np.ndarray): the widths of the neighborhood used for paths to be considered typical for each path length.

            Kwargs:
                paths (np.ndarray/None): the entire sampled path space, the union of the typical and atypical spaces. If not provided these spaces will be merged to generate it.

                p_matrix (np.ndarray/None): the entire matrix of probabilities for each path and each path length. If not provided, this will be generated by merging the p_matrix of the typical and atypical spaces.

                typical_space (InfoSpace/None): the typical set on this space. If None, partitions the provided path space.

                atypical_space (InfoSpace): the atypical set on this space. If None, partitions the provided path space.
        
        """
        # Bool if the space simply needs to be partitioned
        must_partition = (paths is None) or (p_matrix is None)
        # Bool if the space simply needs to be merged since it's already been 
            # partitioned into a typical and atypical space
        must_union = (typical_space is None) or (atypical_space is None)

        if must_partition and must_union:
            # We need either the paths AND the p_matrix or the tupical/atypical 
                # spaces to partition/union the spaces respectively.
            raise TypeError('In sufficient information provided to partition/union the Information Space. We need either paths with their probabilities or the already partitioned spaces.')


        if must_partition:
            # Partition the paths and probability matrix into a typical and
                # atypical space

            # Need to generate the upper/lower bounds for the partitioning
                # of the spaces
            self._lower = entropy_rates - epsilon
            self._upper = entropy_rates + epsilon

            ts_paths = []; ts_p_matrix = []
            ats_paths = []; ats_p_matrix = []

            for path, path_index in enumerate(paths):
                path_prob = p_matrix[path_index]
                # The path entropy rate for direct comparison with the
                    # upper/lower bounds
                path_entropy_rate = -np.log(path_prob[-1]) / path_length

                is_typical = (
                    (self.lower[-1] <= path_entropy_rate)
                    and (path_entropy_rate <= self._upper)
                )

                if is_typical:
                    ts_paths.append(path)
                    ts_p_matrix.append(path_prob)
                else:
                    ats_paths.append(path)
                    ats_p_matrix.append(path_prob)

                typical_space = InfoSpace(ts_paths, ts_p_matrix)
                atypical_space = InfoSpace(ats_paths, ats_p_matrix)

        elif must_union:
            # Union the path data
            ts_empty = (typical_space.paths.size == 0)
            ats_empty = (atypical_space.paths.size == 0)

            if not ts_empty and not ats_empty:
                # Both are nonempty
                paths = np.vstack( (typical_space.paths, atypical_space.paths) )
                p_matrix = np.vstack(
                    (typical_space._p_matrix, atypical_space._p_matrix)
                )
            elif ts_empty:
                # Only the typical_space is empty
                paths = atypical_space.paths
                p_matrix = atypical_space._p_matrix
            else:
                # Only the atypical_space is empty
                paths = typical_space.paths
                p_matrix = typical_space._p_matrix

        ### Storing properties ###
        self._paths = paths
        self._p_matrix = p_matrix

        self._probabilities = self._p_matrix[:, -1]

        self._entropy_rates = entropy_rates

        # Generalize the epsilon to a path_length dependent epsilon for
            # potential generalizations in child classes.
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
            self._upper = self.entropy_rates + self.epsilon        

        return self._upper


    @property
    def lower(self):
        try:
            return self._lower
        except AttributeError:
            # It's never been calculated before.
            self._lower = self.entropy_rates - self.epsilon

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

            path_entropy_rates = -np.log(self._p_matrix) / ns

            self._typicalities = (
                (self.lower <= path_entropy_rates)
                & (path_entropy_rates <= self.upper)
            )

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
        # Hold the current information space to access properties
        old_pinfospace = pinfospace
        # Call parent method
        # Paths and p_matrix will be handled here along with any other 
            # properties shared with parent. Sorted indices of non-degenerate 
            # paths will be calculated here too.
        pinfospace, indices = InfoSpace.shorten(old_pinfospace, path_length, return_index=True)
        
        # Finish the rest of this object's specific properties

        # Truncate the entropy_rates
        entropy_rates = old_pinfospace.entropy_rates[:path_length]

        # Truncate the epsilon
        epsilon = old_pinfospace.epsilon[:path_length]

        # Truncate the typicalities matrix
        # Necessary to re-partition the space.
        # Filter out the typicalities matrix
        typicalities = old_pinfospace.typicalities[indices, :path_length]

        ### Partitioning ###
        ts_paths, ts_p_matrix = [], []
        ats_paths, ats_p_matrix = [], []

        paths = pinfospace.paths
        p_matrix = pinfospace._p_matrix
        for path_index, is_typical in enumerate(typicalities[:, -1]):
            path = paths[path_index]
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

        pinfospace = PartitionedInfoSpace(entropy_rates=entropy_rates, epsilon=epsilon, paths=paths, p_matrix=p_matrix, typical_space=ts, atypical_space=ats)
        
        # Save the pre-generated property
        pinfospace._typicalities = typicalities

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
            entropy_rate(R(i))
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
        upper = entropy_rates + epsilon
        lower = entropy_rates - epsilon

        ts_paths, ts_p_matrix = [], []
        ats_paths, ats_p_matrix = [], []

        path_entropy_rates = -np.log(p_matrix[:, -1]) / path_length

        # Identify the paths that are typical and atypical
        for path_index, path_entropy_rate in enumerate(path_entropy_rates):
            # Can't create typicality matrix since partitioning it will
                # break the ordering
            # Determines whether this path is ultimately typical
            is_typical = (
                (lower[-1] <= path_entropy_rate)
                and (path_entropy_rate <= upper[-1])
            )

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

        pinfospace = PartitionedInfoSpace(
            entropy_rates=entropy_rates,
            epsilon=epsilon,
            paths=paths,
            p_matrix=p_matrix,
            typical_space=ts,
            atypical_space=ats
        )

        # Set pre-calculated properties
        pinfospace._upper = upper
        pinfospace._lower = lower

        return (pinfospace, p_vs_time) if return_p else pinfospace


######################## Entry ########################


def main():
    print('info.py')
    
    ### Testing ###
    p = stp.rand_p(3)
    R = stp.self_assembly_transition_matrix()
    paths = stp.complete_path_space(3, 4)
    pinfospace = PartitionedInfoSpace.partition_space(R, p, paths)
    print( f'pinfospace.total_probability: {pinfospace.total_probability}' )
    print(pinfospace.ats.num_paths)


if __name__ == '__main__':
    main()
