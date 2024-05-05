import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from preprocess import get_data

r2score = tf.keras.metrics.R2Score(class_aggregation="uniform_average", num_regressors=0, name="r2_score")

def lr_schedule(epoch, lr):
    if epoch > 50 and epoch % 20 == 0:
        return lr * 0.9
    return lr

def r2score_fn(y_true, y_pred):
    return r2score(y_true, y_pred)

# data = get_data('black-scholes', 1000000)
data = get_data('heston', 1000000)
X_train, Y_train, X_test, Y_test = data

options_pricing_model = Sequential([
    Dense(512, input_dim=X_train.shape[1], kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(512, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(256, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(256, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(128, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(1, activation='linear')
])

options_pricing_model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mean_squared_error', 
                       r2score_fn
                       ])

csv_logger = tf.keras.callbacks.CSVLogger('heston.log')

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
history = options_pricing_model.fit(X_train, Y_train, epochs=200, batch_size=1024, 
                    validation_split=0.1, callbacks=[lr_scheduler, csv_logger])
test_loss, test_mse, test_r2score = options_pricing_model.evaluate(X_test, Y_test, verbose=2)
predictions = options_pricing_model.predict(X_test)