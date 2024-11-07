import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

#read in commma separated value file with pandas
df = pd.read_csv('/Users/jackson/Documents/GitHub/phys-ga2000/ps_8/survey.csv')

#using data from columns 
age = df["age"]
response = df["recognized_it"]

# logistic function where x is age 
def logistic(x, beta0, beta1):
    return 1 / (1 + np.exp(-(beta0 + beta1 * x)))

# negative log-likelihood
def nlog_likelihood(things):
    beta0, beta1 = things
    probabilities = logistic(age, beta0, beta1)
    likelihood = response * np.log(probabilities) + (1 - response) * np.log(1 - probabilities)
    return -np.sum(likelihood)

#Initial guesses for beta0 and beta1 needed for minimize
initial_guess = [0, 0.25]

#BFGS optimization to find max liklihood estimates for beta0 and beta1
result = minimize(nlog_likelihood, initial_guess)
beta0_mle, beta1_mle = result.x
cov = result.hess_inv  #covariance matrix

#errors are square roots of the diagonal elements of the covariance matrix
errors = np.sqrt(np.diag(cov))

print(f"beta0: {beta0_mle:.4f} ± {errors[0]:.4f}")
print(f"beta1: {beta1_mle:.4f} ± {errors[1]:.4f}")
print("Covariance matrix:")
print(cov)

age_range = np.linspace(min(age), max(age), 100)
predicted_probs = logistic(age_range, beta0_mle, beta1_mle)

plt.scatter(age, response, label="responses (yes = 1, no = 0)")
plt.plot(age_range, predicted_probs, label="logistic model")
plt.xlabel("Age")
plt.ylabel("Probability of yes")
plt.legend()
plt.title("logistic model of sample data")
plt.show()
