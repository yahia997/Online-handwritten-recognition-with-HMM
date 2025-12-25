import pandas as pd
import numpy as np

# forward algorithm ----------------------------------------------------------------
def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
    
    return alpha

# backward algorithm ----------------------------------------------------------------
def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
    
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
    
    return beta

# baum-welch algorithm itself ----------------------------------------------------------------
def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)

    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)
    
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
    
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
    
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
    
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)
    
        b = np.divide(b, denominator.reshape((-1, 1)))
    
    return {"a": a, "b": b, "initial_distribution": initial_distribution}

# organize all in one class ------------------------------------------------------------
class CustomHMM:
    def __init__(self, n_states=5, n_observations=10, n_iter=100):
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_iter = n_iter
        self.A = None
        self.B = None
        self.pi = None
        self.is_trained = False
        
    def fit(self, observations):
        """Train HMM on a single sequence using your Baum-Welch"""
        if len(observations) == 0:
            return
            
        # Initialize parameters randomly
        self.A = np.ones((self.n_states, self.n_states))
        self.A = self.A / np.sum(self.A, axis=1)
        
        self.B = np.ones((self.n_states, self.n_observations))
        self.B = self.B / np.sum(self.B, axis=1).reshape((-1, 1))
        
        self.pi = np.ones(self.n_states) / self.n_states
        
        # Convert observations to discrete indices
        V = np.array(observations, dtype=int)
        
        # Ensure observations are within bounds
        V = np.clip(V, 0, self.n_observations - 1)
        
        # Run Baum-Welch
        try:
            result = baum_welch(V, self.A.copy(), self.B.copy(), self.pi.copy(), self.n_iter)
            self.A = result["a"]
            self.B = result["b"]
            self.pi = result["initial_distribution"]
            self.is_trained = True
        except Exception as e:
            print(f"Baum-Welch training failed: {e}")
            self.is_trained = False
    
    def score(self, observations):
        """Compute log probability of observations given the model"""
        if not self.is_trained or len(observations) == 0:
            return -float('inf')
            
        try:
            V = np.array(observations, dtype=int)
            V = np.clip(V, 0, self.n_observations - 1)
            
            # Use forward algorithm to compute probability
            alpha = forward(V, self.A, self.B, self.pi)
            prob = np.sum(alpha[-1, :])
            
            return np.log(prob) if prob > 0 else -float('inf')
        except:
            return -float('inf')