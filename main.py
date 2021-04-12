#
# нейронная сеть перцепрон
#

import numpy as np

# формула активатор название сигмоидная функция
def sigmoid(x):
    return  1 / (1 + np.exp(-x))

# входные данные
t_inputs = np.array([[0,0,1],
                     [1,1,1],
                     [1,0,1],
                     [0,1,1]])

# ожидаемые вызодные данные
t_outputs = np.array([[0,1,1,0]]).T

# задача рандомных весов сид 1
np.random.seed(345)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

# обучение нейронной сети  20000 раз
for i in range(20000):
    input_layer = t_inputs
    outputs = sigmoid( np.dot(input_layer, synaptic_weights))

    err = t_outputs - outputs
    adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adjustments

print("Think weights:")
print(synaptic_weights)

print("Output:")
print(outputs)

# ТЕСТ

new_inputs = np.array([1, 1, 0])
outputs = sigmoid( np.dot(new_inputs, synaptic_weights))

print("new test")
print(outputs)




