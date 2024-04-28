import numpy as np
import tensorflow as tf
import random
import math
from scipy.stats import norm


def get_data(model, total_samples):
    # suggested total samples is 1,000,000
    data_generator = models[model]
    training_split = int(0.9 * total_samples)
    train_inputs, train_labels = data_generator(training_split)
    test_inputs, test_labels = data_generator(total_samples - training_split)
    return [train_inputs, train_labels, test_inputs, test_labels]

def generate_data_black_scholes(num_samples):
    inputs = np.zeros((num_samples, 4))
    labels = np.zeros((num_samples, 1))

    for i in range(num_samples):
        stock_price = random.uniform(0.5, 1.4) # S_0/K
        time_to_maturity = random.uniform(0.05, 1.0) # tau
        risk_free_rate = random.uniform(0.0, 0.1) # r
        volatility = random.uniform(0.05, 1.0) # sigma
        d1 = (math.log(stock_price) + (risk_free_rate + volatility * volatility * 0.5) * time_to_maturity) / (volatility * time_to_maturity)
        d2 = d1 - volatility * math.sqrt(time_to_maturity)
        call_price = stock_price * norm.cdf(d1) - math.exp(-1 * risk_free_rate * time_to_maturity) * norm.cdf(d2)
        scaled_time_value = call_price - max(stock_price - math.exp(-1 * risk_free_rate * time_to_maturity, 0))
        input = [stock_price, time_to_maturity, risk_free_rate, scaled_time_value]
        label = [volatility]
        inputs[i] = np.array(input)
        labels[i] = np.array(label)
    
    return [inputs, labels]

def generate_data_heston(num_samples):
    inputs = np.zeros((num_samples, 8))
    labels = np.zeros((num_samples, 1))

    for i in range(num_samples):
        strike_price = 1 # K
        moneyness = random.uniform() # m = S_0/K
        time_to_maturity = random.uniform() # tau
        risk_free_rate = random.uniform() # r
        correlation = random.uniform() # rho
        reversion_speed = random.uniform() # kappa
        long_average_variance = random.uniform() # v_bar
        volatility_of_volatility = random.uniform() # gamma
        initial_variance = random.uniform() # v_0
        european_call_price = random.uniform() # V

        input = [moneyness, time_to_maturity, risk_free_rate, correlation, reversion_speed, long_average_variance, volatility_of_volatility, initial_variance]
        label = [european_call_price]
        inputs[i] = np.array(input)
        labels[i] = np.array(label)
    
    return [inputs, labels]

models = {
    'black-scholes': generate_data_black_scholes,
    'heston': generate_data_heston
}