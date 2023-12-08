import numpy as np
import matplotlib.pyplot as plt
import math as math
from sympy import symbols
# Define the function
input = 15 # Use for A
x = symbols('x')
def f(x,A):
    return x**2 -(2*A)*x - A**2
def g(x,A):
    return -x**2 +(4*A)*x + A**3

result_a = 2
result_b = -6*input
result_c = -input**2 -input**3
result_equation = result_a * x**2 + result_b * x + result_c 
d = (result_b**2) - (4*result_a*result_c)
root_1= (-result_b-math.sqrt(d))/(2*result_a)
root_2= (-result_b+math.sqrt(d))/(2*result_a)  
root_3 = f(root_1,input)
root_4 = f(root_2, input)
print("Intersaction point 1 is ",( root_1, root_3))
print("Intersaction point 2 is ",(root_2, root_4))
# Generate x values
xvalues = np.linspace(-1000,1001)   


# Evaluate the function for each x value
fvalues = f(xvalues,input)
gvalues = g(xvalues,input)
# Plot the function
plt.plot(xvalues,fvalues, label='f(x)')
plt.plot(xvalues,gvalues, label='g(x)')
plt.scatter(root_1,root_3, color='red',label= ' Intersection Points')
plt.scatter(root_2,root_4, color='red')


# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Question 1a") 

# Show the plot
plt.show()
plt.grid()