import numpy as np 

def l2_error(pred,y):
    # y => [ntimestep, nmodes]
    up =  np.mean(np.sqrt((pred -y)**2),1)
    down = np.mean(np.sqrt(y**2),1)
    res = up/down
    
    return res 