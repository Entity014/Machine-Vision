import numpy as np
x1 = x2 = -0.5
u1 = -1.3
u2 = -0.9
sigma1 = 0.1
sigma2 = 0.4

fx1 = 1 / np.sqrt(2 * np.pi * pow(sigma1,2))
fy1 = fx1 * np.exp(- (pow(x1- u1, 2) / (2 * pow(sigma1, 2))))
fx2 = 1 / np.sqrt(2 * np.pi * pow(sigma2,2))
fy2 = fx2 * np.exp(- (pow(x2- u2, 2) / (2 * pow(sigma2, 2))))
print(fy1)
print(fy2)
print(fy1 * fy2)