import pdb
from numpy import *
from pymc import *

# Import data
household_file = open('household.csv')
household_names = [eval(x) for x in household_file.next().split(",")]
household_data = array([[eval(x) for x in y.split(",")] for y in household_file])
household, household_county, radon, basement = transpose(household_data)
household_county = household_county.astype(int) - 1
radon = radon.astype(float)
basement = basement.astype(bool)

county_file = open('county.csv')
county_names = [eval(x) for x in county_file.next().split(",")]
county_data = array([[eval(x) for x in y.split(",")] for y in county_file], float)
county, u = transpose(county_data)

# County-level hyperpriors
gamma0 = Uniform('gamma0', -100, 100)
gamma1 = Uniform('gamma1', -100, 100)
sigma_a = Uniform('sigma_a', 0, 100)
tau_a = Lambda('tau_a', lambda s=sigma_a: s**-2)

# County means
mu_a = Lambda('mu_a', lambda g0=gamma0, g1=gamma1: g0 + g1*u)

# County intercepts
alpha = Normal('alpha', mu=mu_a, tau=tau_a)

# Household-level hyperpriors
beta = Uniform('beta', -100, 100)
sigma_y = Uniform('sigma_y', 0, 100)
tau_y = Lambda('tau_y', lambda s=sigma_y: s**-2)

# Household mean radon
mu_y = Lambda('mu_y', lambda a=alpha, b=beta: a[household_county] + b*basement)

y = Normal('y', mu=mu_y, tau=tau_y, value=radon, observed=True)
