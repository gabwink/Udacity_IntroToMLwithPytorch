import matplotlib.pyplot as plt
import numpy as np
def absolute_trick(w1 , w2, x, y, alpha = 0.1):
    """
    point above line y = (w1 + p * alpha ) x + (w2 + alpha )
    point below line y = (w1 - p * alpha ) x + (w2 - alpha )
    """
    q_prime =  (w1 * x) + w2
    p = x
    if q_prime < y:        # point is above
        nw1 = (w1 + (p * alpha) )
        nw2 = (w2 + alpha )
    else:               # point is below
        nw1 = (w1 - (p * alpha) )
        nw2 = (w2 - alpha )
    print(nw1)
    print(nw2)
    return   nw1 * x + nw2

def square_trick(w1 , w2, x, y, alpha = 0.01):
    q_prime =  (w1 * x) + w2
    p = x
    print((w1 + p * (y-q_prime) * alpha))
    print((w2 + (y - q_prime) * alpha))
    return (w1 + p * (y-q_prime) * alpha) * x + (w2 + (y - q_prime) * alpha)

def mean_absolute_error(m, b, points):
    total = 0
    for x,y in points:
        prediction = m * x + b
        total += abs(y - prediction)
    return total/len(points) 

def mean_square_error(m, b, points):
    total = 0
    for x,y in points:
        prediction = m * x + b
        total += (y - prediction)**2
    
    return total/(len(points) *2) 




points = [(2,-2),(5,6),(-4,-4),(-7,1),(8,14)]
print(mean_absolute_error(1.2, 2, points))
print(mean_square_error(1.2, 2, points))