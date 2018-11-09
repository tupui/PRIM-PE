import numpy as np
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from batman.space import Space
from batman.visualization import doe, response_surface, reshow
from batman.functions import Branin
import openturns as ot

# Problem definition: f(sample) -> data
corners = np.array([[-5, 0], [10, 14]])
sample = Space(corners)
sample.sampling(20)

doe(sample, fname='init_doe.pdf')

fun_branin = Branin()
def fun(x): return - fun_branin(x)
data = fun(sample)

# Algo


def random_uniform_ring(center=np.array([0, 0]), r_outer=1, r_inner=0, n_samples=1):
    """Generate point uniformly distributed in a ring.

    Muller, M. E. "A Note on a Method for Generating Points Uniformly on
    n-Dimensional Spheres." Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.
    """
    center = np.asarray(center).reshape(1, -1)
    nd = center.shape[1]
    x = np.random.normal(size=(n_samples, nd))

    # dists = [ot.Normal(0, 1) for _ in range(nd)]
    # dists = ot.ComposedDistribution(dists)
    # lhs = ot.LHSExperiment(dists, n_samples, True, True)
    # x = np.array(ot.SimulatedAnnealingLHS(lhs, ot.GeometricProfile(),
    #                                       ot.SpaceFillingC2()).generate())
    # x = np.array(ot.LowDiscrepancyExperiment(ot.SobolSequence(), dists, n_samples).generate())

    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]  # generate on unit sphere

    # using the inverse cdf method
    # u = np.random.uniform(size=(n_samples))
    u = np.array(ot.LHSExperiment(ot.Uniform(0, 1),
                                  n_samples, True, True).generate()).flatten()

    # this is inverse the cdf of ring volume as a function of radius
    sc = (u * (r_outer ** nd - r_inner ** nd) + r_inner ** nd) ** (1. / nd)
    return x * sc[:, None] + center


# Parameters
threashold = -20

# n_samples/iter = (n_success + n_failure) * n_resamples
n_iterations = 30
n_resamples = 5
n_success = 4
n_failure = 2
n_neighbors = 5
min_radius = 0.02

# Scaling space
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(sample)

def scaler_transform(x, bounds=corners):
    return (x - bounds[0]) / (bounds[1] - bounds[0])

def scaler_inverse_transform(x, bounds=corners):
    return x * (bounds[1] - bounds[0]) + bounds[0]

sample_scaled = scaler_transform(sample.space)


for _ in range(n_iterations):
    sample_ = Space(np.array([[0, 1]] * sample.shape[1]).T)

    # filter success/failure based on data >= threashold
    idx_data = np.argsort(data, axis=0).flatten()
    sample_scaled = sample_scaled[idx_data]
    data = data[idx_data]

    limit = np.where(data >= threashold)[0][0]

    sample_failure = sample_scaled[:limit]
    data_failure = data[:limit]
    sample_success = sample_scaled[limit:]
    data_success = data[limit:]

    # density based on the distance of the K-th neighbour
    k_neigh = NearestNeighbors(n_neighbors=5)
    k_neigh.fit(sample_scaled)
    density = k_neigh.kneighbors(return_distance=True)[0][:, -1]

    density_failure = density[:limit]
    density_success = density[limit:]

    # sort success points by highest density and select n1 from
    idx_success = np.argsort(density_success)[-n_success:]
    sample_success = sample_success[idx_success]
    density_success = density_success[idx_success]

    # random filtering and sort by lowest density and select n2 from
    bounds = [min(len(sample_failure), n_failure), len(sample_failure)]
    n_failure_ = np.random.randint(*bounds)  # number to sort density from
    idx_failure = np.random.randint(0, len(sample_failure), size=n_failure_)
    idx_failure = np.unique(idx_failure)  # idx sample to sort density from

    sample_failure = sample_failure[idx_failure]
    density_failure = density[idx_failure]

    idx_failure = np.argsort(density_failure)[:bounds[0]]
    sample_failure = sample_failure[idx_failure]
    density_failure = density_failure[idx_failure]

    # sample around success and failure samples
    for s, r in zip(sample_success, density_success):
        r = r if r > min_radius else min_radius
        sample_ += random_uniform_ring(center=s, r_outer=r, n_samples=n_resamples)

    for s, r in zip(sample_failure, density_failure):
        r = r if r > min_radius else min_radius
        sample_ += random_uniform_ring(center=s, r_outer=r, n_samples=n_resamples)

    sample_scaled = np.concatenate([sample_scaled, sample_])
    data = np.concatenate([data, fun(scaler_inverse_transform(sample_))])


sample.empty()
sample += scaler_inverse_transform(sample_scaled)
doe(sample, fname='final_doe.pdf')


# Analysis
print(f'\n########### N-samples ###########\n {sample.shape}')

# Filtering in/out
mask_threashold = data >= threashold
mask_threashold = mask_threashold.flatten()
inv_mask_threashold = np.logical_not(mask_threashold)

sample_in = sample[mask_threashold]
sample_out = sample[inv_mask_threashold]

ratio = len(sample_in) / len(sample) * 100

print(f'{ratio:.2f}% of sampling is inside')

####### Visualization #######
fig = response_surface(corners, fun=fun, ticks_nbr=20, contours=[-20])
fig = reshow(fig)

plt.scatter(sample_in[:, 0], sample_in[:, 1], c='r')
plt.scatter(sample_out[:, 0], sample_out[:, 1], c='b')

plt.show()
