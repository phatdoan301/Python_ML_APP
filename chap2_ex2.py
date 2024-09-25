import pandas as pd
import numpy as np

data = pd.read_csv('Practice2_Chapter2.csv')

x = []
x.append(data['TV'])
x.append(data['Radio'])
x.append(data['Newspaper'])
y = data['Sales']

def predict(x, Theta0, Theta1, Theta2, Theta3):
    return Theta0 + Theta1 * x[0] + Theta2 * x[1] + Theta3 * x[2]

def cost(x, y, Theta0, Theta1, Theta2, Theta3):
    m = len(x[0])
    return (1/2*m) * np.sum((predict(x, Theta0, Theta1, Theta2, Theta3) - y)**2)

def gradient_descent(x, y, Theta0, Theta1, Theta2, Theta3, learning_rate):
    m = len(x[0])
    Theta0_new = Theta0 - learning_rate * (1/m) * np.sum(predict(x, Theta0, Theta1, Theta2, Theta3) - y)
    Theta1_new = Theta1 - learning_rate * (1/m) * np.sum((predict(x, Theta0, Theta1, Theta2, Theta3) - y) * x[0])
    Theta2_new = Theta2 - learning_rate * (1/m) * np.sum((predict(x, Theta0, Theta1, Theta2, Theta3) - y) * x[1])
    Theta3_new = Theta3 - learning_rate * (1/m) * np.sum((predict(x, Theta0, Theta1, Theta2, Theta3) - y) * x[2])
    
    Theta0 = Theta0_new
    Theta1 = Theta1_new
    Theta2 = Theta2_new
    Theta3 = Theta3_new
    return Theta0, Theta1, Theta2, Theta3

Theta0 = 0.5
Theta1 = 0.5
Theta2 = 0.5
Theta3 = 0.5
learning_rate = 1e-6
for i in range(0,30):
    Theta0, Theta1, Theta2, Theta3 = gradient_descent(x, y, Theta0, Theta1, Theta2, Theta3, learning_rate)

print(f"{Theta0:.6f} + {Theta1:.6f}x1 + {Theta2:.6f}x2 + {Theta3:.6f}x3")


