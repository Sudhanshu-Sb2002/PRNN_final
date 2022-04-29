import numpy as np
def parzen_window_estimate(x_train,y_train,x_test,h=0):
    if h == 0:
        h=(x_train.size[0])**0.5
    y_test=np.zeros(x_test.size[0])
    windows=np.zeros(int(x_train.size[1]))

def window_functions(dims,*,model='cube',h=2):
    if model=='cube':
        centres=np.array(lins)
    elif model=='gaussian':
        return np.exp(-((x-centre)**2)/(2*h**2))