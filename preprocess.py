import numpy as np
import tensorflow as tf
import random
import math
from scipy.stats import norm
from scipy.integrate import quad
import cmath
import time

i = complex(0, 1)

def get_data(model, total_samples):
    # suggested total samples is 1,000,000
    data_generator = models[model]
    training_split = int(0.9 * total_samples)
    train_inputs, train_labels = data_generator(training_split)
    print("Training data generated.")
    test_inputs, test_labels = data_generator(total_samples - training_split)
    print("Testing data generated.")
    return [train_inputs, train_labels, test_inputs, test_labels]

def generate_data_black_scholes(num_samples):
    inputs = np.zeros((num_samples, 4))
    labels = np.zeros((num_samples, 1))

    ten_percent = int(num_samples / 10)
    percentile = 0

    for j in range(num_samples):
        start_time = time.time()
        stock_price = random.uniform(0.5, 1.4) # S_0/K
        time_to_maturity = random.uniform(0.05, 1.0) # tau
        risk_free_rate = random.uniform(0.0, 0.1) # r
        volatility = random.uniform(0.05, 1.0) # sigma
        d1 = (math.log(stock_price) + (risk_free_rate + volatility * volatility * 0.5) * time_to_maturity) / (volatility * time_to_maturity)
        d2 = d1 - volatility * math.sqrt(time_to_maturity)
        call_price = stock_price * norm.cdf(d1) - math.exp(-1 * risk_free_rate * time_to_maturity) * norm.cdf(d2)
        scaled_time_value = call_price - max(stock_price - math.exp(-1 * risk_free_rate * time_to_maturity), 0)
        input = [stock_price, time_to_maturity, risk_free_rate, scaled_time_value]
        label = [volatility]
        inputs[j] = np.array(input)
        labels[j] = np.array(label)
        if j == 1:
            print(time.time() - start_time)
        # if j%ten_percent == 0:
        #     print(str(percentile) + "% done generating data")
        #     percentile += 10
    
    return [inputs, labels]

def chi_integrand(y, k, a, b):
    return math.exp(y) * math.cos(k * math.pi * ((y - a)/(b - a)))

def chi_k(k, a, b, c, d):
    multiplier = 1 / (1 + ((k * math.pi) / (b - a)) ** 2)
    sin_multiplier = k * math.pi / (b - a)
    term1 = math.cos(k * math.pi * (d - a) / (b - a)) * math.exp(d) 
    term2 = math.cos(k * math.pi * (c - a) / (b - a)) * math.exp(c) 
    term3 = sin_multiplier * math.sin(k * math.pi * (d - a) / (b - a)) * math.exp(d)
    term4 = sin_multiplier * math.sin(k * math.pi * (c - a) / (b - a)) * math.exp(c)

    return multiplier * (term1 - term2 + term3 - term4)

def psi_integrand(y, k, a, b):
    return math.cos(k * math.pi * ((y - a)/(b - a)))

def psi_k(k, a, b, c, d):
    if k == 0:
        return d - c
    else:
        return (math.sin(k * math.pi * (d - a) / (b - a)) - math.sin( k * math.pi * (c - a) / (b - a))) * (b - a) / (k * math.pi)

def phi_heston(omega, u_0, lambda_heston, u_bar, n, delta_t, rho, mu):
    D = cmath.sqrt((lambda_heston - i * rho * n * omega) ** 2 + (omega ** 2 + i * omega) * n * n)
    G = (lambda_heston - i * rho * n * omega - D) / (lambda_heston - i * rho * n * omega + D)
    e1 = cmath.exp(i * omega * mu * delta_t + (u_0/(n ** 2)) * ((1 - cmath.exp(-1 * D * delta_t)) / (1 - G * cmath.exp(-1 * D * delta_t))) 
                   * (lambda_heston - i * rho * n * omega - D))
    e2 = (lambda_heston * u_bar / (n ** 2)) * (delta_t * (lambda_heston - i * rho * n * omega - D) 
                                               - 2 * cmath.log10((1 - G * cmath.exp(-1 * D * delta_t)) / (1 - G)))
    return e1 * e2


def generate_data_heston(num_samples):
    inputs = np.zeros((num_samples, 8))
    labels = np.zeros((num_samples, 1))

    ten_percent = int(num_samples / 10)
    percentile = 0

    for j in range(num_samples):
        start_time = time.time()
        # K = 1
        moneyness = random.uniform(0.6, 1.4) # m = S_0/K
        time_to_maturity = random.uniform(0.1, 1.4) # tau
        risk_free_rate = random.uniform(0.0, 0.10) # r
        correlation = random.uniform(-0.95, 0.0) # rho
        reversion_speed = random.uniform(0.0, 2.0) # kappa
        long_average_variance = random.uniform(0.0, 0.5) # v_bar
        volatility_of_volatility = random.uniform(0.0, 0.5) # gamma
        initial_variance = random.uniform(0.05, 0.5) # v_0
        european_call_price = random.uniform(0.0, 0.67) # V

        l_cos = 50
        n_cos = 1500

        x = math.log(moneyness)
        y = math.log(moneyness)

        ten_percent = int(num_samples / 10)

        c1 = long_average_variance + (1 - math.exp(-1 * reversion_speed) * (long_average_variance - initial_variance)/(2 * reversion_speed) - 0.5 * long_average_variance)
        c2 = (1/(8 * (reversion_speed ** 3))) * (volatility_of_volatility * reversion_speed * math.exp(-1 * reversion_speed) * (initial_variance - long_average_variance) * (8 * reversion_speed * correlation - 4 * volatility_of_volatility)
            + reversion_speed * correlation * volatility_of_volatility * (1 - math.exp(-1 * reversion_speed)) * (16 * long_average_variance - 8 * initial_variance)
            + 2 * long_average_variance * reversion_speed * (-4 * reversion_speed * correlation * volatility_of_volatility + volatility_of_volatility ** 2 + 4 * reversion_speed * reversion_speed)
            + (volatility_of_volatility ** 2) * ((long_average_variance - 2 * initial_variance) * math.exp(-2 * reversion_speed) + long_average_variance * (6 * math.exp(-1 * reversion_speed) - 7) + 2 * initial_variance)
            + 8 * (reversion_speed ** 2) * (initial_variance - long_average_variance) * (1 - math.exp(-1 * reversion_speed))
            )

        a = c1 - 12 * math.sqrt(abs(c2))
        b = c1 + 12 * math.sqrt(abs(c2))

        complex_sum = 0
        for k in range(n_cos):
            phi = phi_heston(k * math.pi / (b - a), initial_variance, reversion_speed, long_average_variance, volatility_of_volatility, time_to_maturity, correlation, long_average_variance)
            U_k = (2 / (b - a)) * (chi_k(k, a, b, 0, b) - psi_k(k, a, b, 0, b))
            complex_sum += phi * U_k * cmath.exp(i * k * math.pi * ((x - a) / (x - b)))          

        european_call_price = math.exp(-1 * risk_free_rate * time_to_maturity) * complex_sum.real

        input = [moneyness, time_to_maturity, risk_free_rate, correlation, reversion_speed, long_average_variance, volatility_of_volatility, initial_variance]
        label = [european_call_price]

        inputs[j] = np.array(input)
        labels[j] = np.array(label)

        if j == 1:
            print(time.time() - start_time)

        if j%ten_percent == 0:
            print(str(percentile) + "% done generating data")
            percentile += 10
    
    return [inputs, labels]

models = {
    'black-scholes': generate_data_black_scholes,
    'heston': generate_data_heston
}