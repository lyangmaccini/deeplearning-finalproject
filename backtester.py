import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from preprocess import get_data

class OptionsPricingModel:
    def __init__(self, input_dim):
        self.model = self.create_model(input_dim)

    def create_model(self, input_dim):
        model = Sequential()
        model.add(Dense(512, input_dim=input_dim, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(512, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(256, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(256, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(128, kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(1, activation='linear'))

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return model

    def fit(self, X, Y, epochs, batch_size, validation_split, callbacks):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, 
                       validation_split=validation_split, callbacks=callbacks)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        errors = predictions - Y
        mse = np.mean(errors ** 2)
        return mse

    def backtest(self, X_test, Y_test):
        mse = self.evaluate(X_test, Y_test)
        print(f"Backtest MSE: {mse}")
    
def lr_schedule(epoch, lr):
    if epoch > 50 and epoch % 20 == 0:
        return lr * 0.9
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
data = get_data('black-scholes', 1000000)
X_train, Y_train, X_test, Y_test = data
options_pricing_model = OptionsPricingModel(X_train.shape[1])
options_pricing_model.fit(X_train, Y_train, epochs=200, batch_size=1024, 
                          validation_split=0.1, callbacks=[lr_scheduler])
options_pricing_model.backtest(X_test, Y_test)