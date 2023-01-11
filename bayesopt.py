
from itertools import combinations
import numpy as np
import torch
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf

num_restarts = 3
num_raw_samples = 128

class ActivePreferenceLearning:
    """
    Generic model to learn user preferences
    """

    def __init__(self, bounds, init_X, q=4):

        self.bounds = torch.tensor(bounds)

        self.X = init_X
        self.bounds = self.bounds
        self.D = torch.LongTensor([])
        self.q = q

        self.trials = 0

        self.model = None
        self.mll = None

    def acquire(self):
        acq_func = qNoisyExpectedImprovement(
            model=self.model,
            X_baseline=self.X.float()  # Already observed
        )
        # optimize and get new observation
        next_X, acq_val = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.q,
            num_restarts=num_restarts,
            raw_samples=num_raw_samples,
        )

        return next_X.to(self.X)

    def update_data(self, next_X, next_D):

        if len(self.D) == 0:
            self.D = torch.cat([self.D, torch.LongTensor(next_D)])
        else:
            self.D = torch.cat([self.D, torch.LongTensor(np.array(next_D) + self.X.shape[-2])])
            self.X = torch.cat([self.X, next_X])

        self.fit()

        return self.top()

    def fit(self):
        # refit models
        if len(self.D) > 0:
            self.model = PairwiseGP(self.X, self.D, outcome_transform=Standardize(m=1))

            mll = PairwiseLaplaceMarginalLogLikelihood(self.model.likelihood, self.model)
            self.mll = fit_gpytorch_model(mll)

    def top(self):
        return self.X[self.model.utility.argmax()].tolist()


if __name__=='__main__':
    # data generating helper functions
    def utility(X):
        """Given X, output corresponding utility (i.e., the latent function)"""
        # y is weighted sum of X, with weight sqrt(i) imposed on dimension i
        weighted_X = X * torch.sqrt(torch.arange(X.size(-1), dtype=torch.float) + 1)
        y = torch.sum(weighted_X, dim=-1)
        return y


    def generate_data(n, dim=2):
        """Generate data X and y"""
        # X is randomly sampled from dim-dimentional unit cube
        # we recommend using double as opposed to float tensor here for
        # better numerical stability
        X = torch.rand(n, dim, dtype=torch.float64)
        y = utility(X)
        return X, y


    def generate_comparisons(y, n_comp, noise=0.1, replace=False):
        """Create pairwise comparisons with noise"""
        # generate all possible pairs of elements in y
        all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
        # randomly select n_comp pairs from all_pairs
        comp_pairs = all_pairs[np.random.choice(range(len(all_pairs)), n_comp, replace=replace)]
        # add gaussian noise to the latent y values
        c0 = y[comp_pairs[:, 0]] + np.random.standard_normal(len(comp_pairs)) * noise
        c1 = y[comp_pairs[:, 1]] + np.random.standard_normal(len(comp_pairs)) * noise
        reverse_comp = (c0 < c1).numpy()
        comp_pairs[reverse_comp, :] = np.flip(comp_pairs[reverse_comp, :], 1)
        comp_pairs = torch.tensor(comp_pairs).long()

        return comp_pairs

    q = 4
    dim = 5
    q_comp = 1
    noise = 0.1

    NUM_BATCHES = 10

    torch.manual_seed(0)
    np.random.seed(0)
    best_vals = []

    # Create initial data
    init_X, init_y = generate_data(q, dim=dim)
    init_D = generate_comparisons(init_y, q_comp, noise=noise)

    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

    # Create model
    learner = ActivePreferenceLearning(bounds, init_X=init_X, q=q)

    top = learner.update_data(init_X, init_D)

    best_vals.append(top)

    for i in range(NUM_BATCHES):

        next_X = learner.acquire()
        next_D = generate_comparisons(utility(next_X), q_comp, noise=noise)

        top = learner.update_data(next_X, next_D)

        best_vals.append(top)

    print(best_vals)