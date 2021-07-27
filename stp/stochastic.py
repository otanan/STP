#!/usr/bin/env python3
"""Handles generating random objects to use in various calculations.

**Author: Jonathan Delgado**

"""

######################## Imports ########################


import sys
import numpy as np
import scipy.linalg
import bisect # for binary search

# For ProgressBar
import stp.tools.gui as gui


# The random number generator to be used
# _RNG = np.random.default_rng(seed=0) # Seeded one for testing
_RNG = np.random.default_rng()
# We may want to replace zeros by machine epsilon to avoid division by zero 
# errors when calculating quantities of interest
MACHINE_EPSILON = np.finfo(float).eps


######################## Constructors ########################


def rand_p(n=2, zeros=0):
    """ Creates a random probability distribution, currently implemented only 
        with uniform sampling.
    
        Kwargs:
            n (int): the dimension of the desired distribution
    
            zeros (int): the number of desired zeros to be injected into the distribution. Allows for ease of testing edge cases.
    
        Returns:
            (np.ndarray): the random distribution
    
    """
    # Error checking for input amounts
    if zeros > n - 1:
        print('Too many zeros, will not allow for normalization.')
        sys.exit()
    elif zeros < 0:
        print('Invalid amount of zeros. Should be nonnegative.')
        sys.exit()
    # Creates a random distribution of length n
    p = _RNG.random(n)
    # Set the first m entries equal to 0, where m is the amount of zeros 
        # requested
    if zeros != 0:
        p[:zeros] = 0
        # Shuffle the contents to avoid just having the same first entries as 0
        np.random.shuffle(p)
    # Return the distribution normalized by its 1-norm    
    return p / np.sum(p)


def rand_rate_matrix(n=2, seed=None):
    """ Generates a random, time-independent, n x n rate matrix consisting of 
        probabilities per unit time. Column normalized.
    
        Kwargs:
            n (int): the number of states the matrix will correspond to. Relates to the dimensions of the matrix.

            seed (None/int): the seed for sampling. Unseeded (uses module-level rng) if None.
    
        Returns:
            (np.ndarray): the rate matrix.
    
    """
    # Use module-level rng if no seed is provided, else, seed one
    rng = _RNG if seed is None else np.random.default_rng(seed=seed)

    # Generates a random n x n matrix
    W = rng.random( (n, n) )

    np.fill_diagonal(W, 0)
    # Normalize the columns to sum to zero.
    for y in range(n):
        W[y, y] = -np.sum(W[:,y])

    return W


def rate_to_transition_matrix(W, time_step):
    """ Converts a rate matrix to a transition matrix assuming a constant 
        control parameter during the duration of time_step.
        
        Args:
            W (np.ndarray): the rate matrix

            time_step (float): the time step
    
        Returns:
            (np.ndarray): the transition matrix
    
    """
    return scipy.linalg.expm(W * time_step)


def rand_transition_matrix(n=2, time_step=1.0):
    """ Generates a random, time-independent, discrete time, transition matrix
        by first generating a random rate matrix and then matrix exponentiating it to incorporate the time step as an additional parameter.
    
        Kwargs:
            n (int): the number of states the matrix will correspond to. Relates to the dimensions of the matrix.

            time_step (float): the time step: the interval of time between observations.
    
        Returns:
            (np.ndarray): the n x n transition matrix
    
    """
    return rate_to_transition_matrix(rand_rate_matrix(n), time_step)


def self_assembly_rate_matrix(energy=1, c=1, M=1, T=1):
    """ Generates a self-assembly rate matrix for a 3-state system following:
        https://aip.scitation.org/doi/10.1063/1.3662140.
        
        Kwargs:
            energy (float): the negative of the "optimally bound level of energy"

            c (float): "concentration-like variable"

            M (int): the degeneracy of the misbound level

            T (float/function): the temperature as a float or a function which returns the temperature as a function of time. In the latter case the returned rate matrix will be a function which provides a np.ndarray as a function of time.
    
        Returns:
            (np.ndarray/function): the time-independent rate matrix as a numpy array in the case where the temperature is constant. Otherwise returns a function corresponding to the time-dependent rate matrix.
    
    """
    if not isinstance(M, int):
        print('The degeneracy of the misbound level, M, must be an integer.')
        sys.exit()

    if not callable(T) and T <= 0:
        print('Must provide a nonnegative temperature.')
        sys.exit()

    if callable(T):
        # T is a time-dependent temperature
        # Provide the time-dependent rate matrix
        return lambda t : self_assembly_rate_matrix(energy=energy, c=c, M=M, T=T(t))

    ### Main body ###

    alpha = np.exp(-energy / (2 * T))
    W = np.array([
        [   -c * (M + 1),   alpha,      alpha**2    ],
        [   c * M,          -alpha,     0           ],
        [   c,              0,          -alpha**2   ]
    ])

    return W


def self_assembly_transition_matrix(energy=1, c=1, M=1, T=1, time_step=1):
    """ Generates a self-assembly, discrete time, transition matrix for a 
        3-state system following: https://aip.scitation.org/doi/10.1063/1.3662140. Done assuming any external control parameter is fixed for the duration of the time_step. This matrix is step dependent, so conversions will be done to time to calculate the current temperature.
    
        Kwargs:
            energy (float): the negative of the "optimally bound level of energy"

            c (float): "concentration-like variable"

            M (int): the degeneracy of the misbound level

            T (float/function): the temperature as a float or a function which returns the temperature as a function of time. In the latter case the returned transition matrix will be a function which provides a np.ndarray as a function of time. This external control parameter will be fixed for the duration of the time_step.

            time_step (float): the time step
    
    
        Returns:
            (np.ndarray/function): the transition matrix
    
    """
    if callable(T):
        # Convert n to time.
        # This also serves to fix the temperature between observations to change
            # the control parameter discretely.
        R = lambda n : self_assembly_transition_matrix(energy=energy, c=c, M=M, T=T(n * time_step), time_step=time_step)
        return R

    return rate_to_transition_matrix(
            self_assembly_rate_matrix(energy=energy, c=c, M=M, T=T),
            time_step
        )


######################## Probability Operations ########################


def step(matrix, p):
    """ Evolves a probability distribution one step forward by computing the 
        matrix multiplication between matrix and p. In the case of the matrix being a rate matrix the output is the time-derivative of p.
        
        Args:
            matrix (np.ndarray): the transition or rate matrix

            p (np.ndarray): the marginal distribution
    
        Returns:
            (np.ndarray): the evolved marginal
    
    """
    return np.matmul(matrix, p)
    

def get_stationary_distribution(matrix, discrete=True):
    """ Calculates the stationary distribution of a transition or rate matrix. 
        Credit: https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain.
        
        Args:
            matrix (np.ndarray/function): the transition or rate matrix
    
        Kwargs:
            discrete (bool): True if the provided matrix is a discrete time transition matrix, False if the provided matrix is a continuous time rate matrix
    
        Returns:
            (np.ndarray/function): the limiting distribution (as a function of time in the case where the matrix is too)
    
    """
    if callable(matrix):
        # Return the limiting distribution as a function of time
        return lambda t : get_stationary_distribution(matrix(t), discrete=discrete)

    if not discrete:
        # Use the simplified method for the rate matrix, not as robust
            # Make sure to check output
        _, eigenvec = np.linalg.eig(matrix)
        print('Returning experimental right eigenvector to the rate matrix. Always check the vector is correct!')
        return np.abs(eigenvec[:, 0])

    S, U = np.linalg.eig(matrix)
    # Added the need to take the real part to avoid +0j added to each term
    # and raising a complex error
    # Added absolute value since I was occasionally getting the negative p
    stationary = np.absolute(
        np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat).real
    )
    return stationary / np.sum(stationary)


def get_path_probability(R, p, path):
    """ Calculates the probability of observing a provided path using the a
        (potentially time-inhomogeneous) transition matrix, the initial marginal distribution, and the path itself.
        
        Args:
            R (function/np.ndarray): the transition matrix. A function of the observation step if the matrix is time-inhomogeneous. That is, R = R(n). In which case Pr( x <- y ) = R(1)[x,y] * p[y]. Provide a numpy matrix in the case of a time-homogeneous transition matrix.

            p (np.ndarray): the marginal distribution.

            path (list/np.ndarray): the path.
    
        Returns:
            (float): the probability of observing this path.
    
    """
    # Check if R is time-independent, if so, convert it to a function
        # of the observation step that simply returns the constant matrix.
    if not callable(R):
        matrix = R
        R = lambda n: matrix

    # Comprehension to collect transition probabilities for vectorized product
    total_jump_prob = np.array([
        R(i + 1)[path[i+1], path[i]]
        for i in range( len(path) - 1 )
    ]).prod()

    return total_jump_prob * p[path[0]]


######################## Path space sampling ########################


def complete_path_space(n, path_length):
    """ Generates the entire path space as a matrix with each row corresponding 
        to a path and each column corresponding to an observation step.
        
        Args:
            n (int): the number of states in the state space

            path_length (int): the length of each path
    
        Returns:
            (np.ndarray): the entire path space. Will be a matrix of size: n^path_length x path_length.
    
    """
    paths = []

    num_paths = n**path_length

    bar = gui.ProgressBar(num_paths, width=300, title='Generating path space...')
    for i in range(num_paths):
        path = np.base_repr(i, base=n)
        path = ( path_length - len(path) )*'0' + path
        paths.append(
            # Convert the number from decimal to base n.
            # This will be in a string format, turn each digit into a separate
                # int in a list, this will be a path.
                [ int(state) for state in path ]
            )

        bar.update()

    bar.finish()

    return np.array(paths)


def KMC(W, p, num_paths, path_length, time_step=1, seed=None, _degenerate_threshold=0.985):
    """ A Rejection-free Kinetic Monte Carlo (KMC) algorithm for simulating 
        the discrete time evolution of a system, where some processes can occur with known (continuous time) rates W = W(t). The discrete time dynamics will be computed using the continuous time rate matrix, W, by freezing the control parameter for moments of time, time_step. From: https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo.
        
        Args:
            W (np.ndarray/function): rate matrix for np.ndarray. If function is provided then W is the function of time that provides a rate matrix (W = W(t)) for each moment in time. Will be used to implement driven systems. If W is just a rate matrix then W(t) is just the time-homogeneous rate matrix.

            p (np.ndarray): the initial marginal distribution.

            num_paths (int): the number of paths to sample.

            path_length (int): the length of each path.
    
        Kwargs:
            time_step (float): the interval between changing of rate matrix (changing of control parameter), and the delay between observations.

            seed (None/int): the seed for sampling. Unseeded (uses module-level rng) if None.

            _degenerate_threshold (float): the fraction of paths that we require to be unique. If not satisfied KMC will be run again with an increased threshold.
    
        Returns:
            (np.ndarray): matrix of paths sampled via KMC.
    
    """


    #------------- Helper functions -------------#


    def _KMC_jump(cumulatives, state):
        """ Helper: takes cumulatives, a state, and generates 
            the next likely jump via the KMC algorithm and the time it takes to observe the jump. Useful for discretizing the KMC.
            
            Args:
                cumulatives (np.ndarray): the KMC's cumulatives.

                state (int): the current state.
        
            Returns:
                (2-tuple): (next_state, delta time)
        
        """
        # https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo.
        # The cumulative relevant to the current state
        relevant_cumulative = cumulatives[:, state]
        u = rng.uniform(0, 1)
        scaled_total_rate = u * cumulatives[-1, state]
        next_state = bisect.bisect_left(relevant_cumulative, scaled_total_rate)
        u = rng.uniform(0, 1)
        jump_time = -np.log(u) / relevant_cumulative[-1]

        return next_state, jump_time


    #------------- Initialization -------------#

    # Use module-level rng if no seed is provided, else, seed one
    rng = _RNG if seed is None else np.random.default_rng(seed=seed)

    if not callable(W):
        # Hold the rate matrix in memory.
        constW = W
        # Let the time-dependent rate matrix just be the constant matrix.
        W = lambda t: constW

    # Read off the size of the state space.
    n = W(0).shape[0]
    state_options = np.arange(n)
    # Read off the final time for the full time series observations given 
        # the path length and the delay between observations.
    # For final_time, time_step, and path_length, the relationship is:
    # time_step * (number of jumps) = final_time
    final_time = time_step * (path_length - 1)

    # Initialize the matrix storing the paths
    paths = np.full( (num_paths, path_length), -1, dtype='int64' )
    # Set the first column of the matrix to be initial state weighted by p.
    paths[:, 0] = rng.choice(state_options, p=p, size=num_paths)

    # Fill each column
    for k in range(1, path_length):
        # This will find the current time of the control parameter to fix
            # the rate matrix since this is discrete time.
        current_time = k * time_step
        # Fix the rate matrix
        current_W = W(current_time)

        #------------- Main algorithm -------------#

        # We need to remove the normalization for the algorithm and just 
            # look at the positive rates.
        posW = current_W.copy()
        np.fill_diagonal(posW, 0)
        # Matrix of partial sums, last one being the total rate
        # Each row corresponds to the same final index for the partial sum
        # Each column corresponds to which rates were summed together
        cumulatives = np.array( [ [ 
            np.sum(posW[:i+1, x])
            for i in range(n)
        ]   for x in range(n)
        ] ).T

        # Run through the path matrix.
        for path in paths:
            current_state = path[k - 1]
            # The state we are currently in is the last nonnegative 
                # element of our path list (since the path matrix is
                # initialized with -1)
            # current_state = path[path >= 0][-1]
            
            total_jump_times = 0
            # Prepare for the while loop
            next_state = current_state
            # Keep evolving the state until it fits what would be observed
                # in a discrete time process.
            while total_jump_times < time_step:
                potential_jump, jump_time = _KMC_jump(cumulatives, next_state)
                
                # Note that if the next jump takes longer than our 
                    # time_step we won't observe it. Since we will
                    # make the measurement before this happens.
                if total_jump_times + jump_time >= time_step:
                    break

                total_jump_times += jump_time
                next_state = potential_jump

            path[k] = next_state

    ### Degenerate paths check ###
    # Check for degenerate paths and sample again if needed.
    paths = np.unique(paths, axis=0)
    if paths.shape[0] / num_paths < _degenerate_threshold:
        # Generate a rate matrix of perturbations
        deltaW = rand_rate_matrix(n, seed=seed) / 100
        Wprime = lambda t : W(t) + deltaW

        # Run KMC again with an increased degeneracy threshold and 
            # a perturbed rate matrix.
        new_paths = KMC(Wprime, p, num_paths, path_length, time_step=time_step, seed=seed, _degenerate_threshold=_degenerate_threshold-10E-9)

        # Combine the path data
        paths = np.unique( np.vstack((paths, new_paths)), axis=0 )
        
    # Return only the requested number of paths (in case we have extra unique 
        # ones from previous degeneracies)
    return paths[:num_paths]


def direct_sampling(R, p):
    """ Not yet been implemented: Samples the path space directly to reflect 
        the dynamics given by the initial marginal and the transition matrix.
        
        Args:
            R (function/np.ndarray): the (possibly time-dependent) transition matrix.

            p (np.ndarray): the marginal distribution.
     
        Returns:
            (np.ndarray): the sampled portion of the path space.
    
    """
    print('Direct sampling has not been implemented yet.')
    

######################## Entry Code ########################


def main():
    print('stochastic.py')

    W = self_assembly_transition_matrix()
    p = np.array([1 - 2 * MACHINE_EPSILON, MACHINE_EPSILON, MACHINE_EPSILON])

    paths = KMC(W, p, 500, 20, seed=0)
    paths2 = KMC(W, p, 500, 20, seed=0)
    print( f'paths.shape: {paths.shape}' )
    
    print( f'paths: {paths}' )
    print( f'paths - paths2: {paths - paths2}' )
    
    
    
    
if __name__ == '__main__':
    main()