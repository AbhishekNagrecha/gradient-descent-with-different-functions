
# %matplotlib inline
# creating the fn and plotting it

# import inline as inline
# import matplotlib as matplotlib
import numpy as np
import matplotlib.pyplot as draw

fn = lambda t: (t ** 2)   # -(3 *(t ** 2))+7

t = np.linspace(-5,5,500)

def fderivat(t):

    t_fderivat = 2 * (t) # ** 2) - (6 * (t))
    return t_fderivat

def gd_algorithm(t_nvalues, t_prevs, precision, learn_rate):

    t_lst, y_lst = [t_nvalues], [fn(t_nvalues)]
    # we are looping till the time we get the desired precision
    while abs(t_nvalues - t_prevs) > precision:
        t_prevs = t_nvalues
        d_t = - fderivat(t_prevs)
        t_nvalues = t_prevs + (learn_rate * d_t)
        t_lst.append(t_nvalues)
        y_lst.append(fn(t_nvalues))

    print("The Local minima value takes place at this: " + str(t_nvalues))
    print("The Number of steps taken are as follows: " + str(len(t_lst)))

    draw.subplot(1, 2, 2)
    draw.scatter(t_lst, y_lst, c="g")
    draw.plot(t_lst, y_lst, c="g")
    draw.plot(t, fn(t), c="r")
    draw.title("GRADIENT D ALGORITHM")
    draw.show()


gd_algorithm(4.5, 0, 0.001, 0.05)