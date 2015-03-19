
import numpy as np
from numpy.testing import assert_almost_equal
from distance import euclidean_dist

def test_non_negativity():
    u = np.random.normal(3)
    v = np.random.normal(3)
    assert euclidean_dist(u, v) >= 0

def test_coincidence_when_zero():
    u = np.zeros(3)
    v = np.zeros(3)
    assert euclidean_dist(u, v) == 0

def test_coincidence_when_not_zero():
    u = np.random.random(3)
    v = np.zeros(3)
    assert euclidean_dist(u, v) != 0

def test_symmetry():
    u = np.random.random(3)
    v = np.random.random(3)
    assert euclidean_dist(u, v) == euclidean_dist(v, u)

def test_triangle():
    u = np.random.random(3)
    v = np.random.random(3)
    w = np.random.random(3)
    assert euclidean_dist(u, w) <= euclidean_dist(u, v) + euclidean_dist(v, w)

def test_known1():
    u = np.array([0])
    v = np.array([3])
    assert_almost_equal(euclidean_dist(u, v), 3)

def test_known2():
    u = np.array([0,0])
    v = np.array([3, 4])
    assert_almost_equal(euclidean_dist(u, v), 5)

def test_known3():
    u = np.array([0,0])
    v = np.array([-3, -4])
    assert_almost_equal(euclidean_dist(u, v), 5)