"""
Anthony Testa, Bennett Brain

Implementations for our CS6140 course project, fall 2023.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm.autonotebook import tqdm


def PCA(data: np.ndarray, eigen: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA on given data.

    Parameters
    --------
    data : np.ndarray
        Dataset, n samples by m features.
    eigen : bool, default=False
        Return the sorted eigen values and vectors.

    Returns
    --------
    transformed_dataset : np.ndarray
        Transformed dataset.
    sorted_eigen_values : np.ndarray, optional
        Sorted eigen values for features.
    sorted_eigen_vectors : np.ndarray, optional
        Sorted eigen vectors for features.
    """

    # Center the data on the mean
    centered_data = data - np.mean(data, axis=0)
    # Compute the covariance matrix, using 32 bit precision to avoid overflowing
    # Measures the strength and direction of correlation between two features (in each cell of the matrix)
    cov = np.cov(centered_data, ddof=0, rowvar=False, dtype=np.float32)
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # Sort by increasing eigen values
    sorted_eigen_indices = np.argsort(-eigenvalues)
    sorted_eigen_values = eigenvalues[sorted_eigen_indices]
    sorted_eigen_vectors = eigenvectors[:, sorted_eigen_indices]

    # Transform dataset by taking the dot product with the eigen vectors
    transformed_dataset = data @ sorted_eigen_vectors

    return transformed_dataset if not eigen else transformed_dataset, sorted_eigen_values, sorted_eigen_vectors


# A lot of cleaning up to do here, I'll get to it but for now just putting this here
def tSNE(sample) -> np.ndarray:
    """

    Perform tSNE on a given dataset.

    Parameters
    --------
    sample : ndarray
        Input dataset

    Returns
    --------
    Y : ndarray
        Output of tSNE.
    """
    # Prep data
    X = MinMaxScaler().fit_transform(sample)
    N, F = X.shape
    no_dims = 2
    PCA_dims = 50
    use_momentum = 1
    perplexity = 30.0

    # Run PCA
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    X = np.dot(X, M[:, 0:PCA_dims]).real

    # Compute pairwise distance
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    # Compute similarities p_ij for each row i:
    P = np.zeros((n, n))
    for i in tqdm(range(n)):
        beta = np.ones((n, 1))
        D_i = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        P_i = np.zeros((1, n))

        # search for beta(i) so that i has effectively 30 neighbors: H=~log2(30)
        trials = 0
        Hdiff = 1
        betamin = -np.inf
        betamax = np.inf

        while np.abs(Hdiff) > 1e-5 and trials < 50:
            P_i = np.exp(-D_i.copy() * beta[i])
            sumP = sum(P_i)
            H = np.log(sumP) + beta[i] * np.sum(D_i * P_i) / sumP
            P_i = P_i / sumP
            H_diff = H - np.log(perplexity)

            if H_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            P_i = np.exp(-D_i.copy() * beta[i])
            sumP = sum(P_i)
            H = np.log(sumP) + beta[i] * np.sum(D_i * P_i) / sumP
            P_i = P_i / sumP
            H_diff = H - np.log(perplexity)
            trials += 1

        # Write row i
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = P_i

    # make sure P is correct
    P = P + np.transpose(P)  # symmetric
    P = P / np.sum(P)  # normalized
    P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)

    # initialize tSNE
    max_iter = 400
    epsilon = 500
    min_gain = .01
    Y = np.random.randn(n, no_dims)
    y_grads = np.zeros((n, no_dims))
    y_incs = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Run tSNE iterations
    for iteration in range(max_iter):

        sum_y2 = np.sum(np.square(Y), 1)
        Qnum = 1. / (1. + np.add(np.add(-2. * np.dot(Y, Y.T), sum_y2).T, sum_y2))
        Qnum[range(n), range(n)] = 0.
        Q = np.maximum(Qnum / np.sum(Qnum), 1e-12)

        # Compute gradient
        L = P - Q
        for i in range(n):
            y_grads[i, :] = np.sum(np.tile(L[:, i] * Qnum[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        if use_momentum:  # add momentum for grads in prev direction
            gains = (gains + 0.2) * ((y_grads > 0.) != (y_incs > 0.)) + (gains * 0.8) * (
                    (y_grads > 0.) == (y_incs > 0.))
            gains[gains < min_gain] = min_gain
            y_incs = - epsilon * (gains * y_grads)
        else:  # simple gradient descent
            y_incs = - epsilon * y_grads

        # update y positions by gradients
        Y = Y + y_incs
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # print loss and scatterplot every ten iterations
        if (iteration + 1) % 10 == 0:
            cost = np.sum(P * np.log(P / Q))
            print(f'Iteration {iteration + 1}: error is {cost}')
            plt.scatter(Y[:, 0], Y[:, 1])
            plt.show()

        # Stop exaggeration
        if iteration == 100:
            P = P / 4.
    return Y
