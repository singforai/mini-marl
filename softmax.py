import matplotlib.pyplot as plt
import numpy as np

z = [-1.5151, -2.4293, -1.6226, -1.1132, -1.7944]

def softmax(a) : 
    c = np.max(a) 
    exp_a = np.exp(a-c) 
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y



def softmax_with_temperature(z, T) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

print("orginal: ", z)

print(softmax(z))
print(softmax_with_temperature(z, 2))
print(softmax_with_temperature(z, 3))
