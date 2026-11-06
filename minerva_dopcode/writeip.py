import numpy as np

ip = np.load('synthetic_IPs/ipdict.npy')

for i in range(len(ip)):
    f = open(str(ip[i]['wav']) + '_' + str(ip[i]['order']) + '.txt','w')
    for j in range(len(ip[i]['ip'])):
        f.write(str(ip[i]['ip'][j]) + '\n')
    f.close()


