import numpy as np

x = np.array([0.245,0.247,0.285,0.299,0.327,0.347,0.356,
0.36,0.363,0.364,0.398,0.4,0.409,0.421,
0.432,0.473,0.509,0.529,0.561,0.569,0.594,
0.638,0.656,0.816,0.853,0.938,1.036,1.045])
y = np.array([0,0,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,1,
1,1,1,1,1,1,1])

def predict(x, theta0, theta1):
    return 1/(1+np.exp(-(theta0 + theta1*x)))

def cost(x, y, theta0, theta1):
    m = len(x)
    y_pre = predict(x, theta0, theta1)
    epsilon = 1e-15
    return -((1/m) * np.sum(y * np.log(y_pre+epsilon) + (1-y) * np.log(1-y_pre+epsilon)))

def grad_descent(x, y, theta0, theta1, learning_rate):
    m = len(x)
    theta0_new = theta0 - learning_rate * (1/m) * np.sum(predict(x, theta0, theta1) - y)
    theta1_new = theta1 - learning_rate * (1/m) * np.sum((predict(x, theta0, theta1) - y) * x)
    theta0 = theta0_new
    theta1 = theta1_new
    return theta0, theta1

theta0 = np.random.rand()
theta1 = np.random.rand()
learning_rate = np.random.rand()

for i in range(0,50):
    theta0, theta1 = grad_descent(x, y, theta0, theta1, learning_rate)
    
print(theta0, theta1, cost(x, y, theta0, theta1))