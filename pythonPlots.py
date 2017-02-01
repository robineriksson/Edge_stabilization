#plot results from the convergences.
import matplotlib.pyplot as plt
import numpy as np

h = np.array([1./4, 1./8, 1./16, 1./32])
E = np.array([0.249538747793,0.20933984302,0.174346095067,0.146470450084])
pE = np.polyfit(h,E,1)
plt.loglog(h,E,'*')
plt.ylabel('L2 norm of error')
plt.xlabel('mesh length h')
plt.legend('p is ?')
plt.show()
#[ 0.45075504  0.14210093]
