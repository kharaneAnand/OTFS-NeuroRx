import numpy as np
from OTFSResGrid import OTFSResGrid
from OTFS import OTFS
import matplotlib.pyplot as plt

# batch
batch_size = 10;

# configuration
N = 15;     # timeslote number
M = 16;     # subcarrier number
QAM = 4;
constel = [-0.7071-0.7071j, -0.7071+0.7071j, 0.7071-0.7071j, 0.7071+0.7071j];
SNR_d = np.random.choice(np.arange(1, 31), size=batch_size, replace=False);
No = 10**(-SNR_d/10);
Es_d = 1;

# channel configuration
p = 6;
lmax = 3;
kmax = 5;

'''
init generalised variables
'''
# OTFS module
otfs = OTFS(batch_size=batch_size);


'''
Tx
'''
# generate symbols
sym_idx = np.random.randint(4, size=(batch_size, M*N));
syms_vec = np.take(constel,sym_idx);
syms_mat = np.reshape(syms_vec, [batch_size, N, M]);
# generate X_DD
X_DD = syms_mat;
# generate
rg = OTFSResGrid(M, N, batch_size=batch_size);
rg.setPulse2Recta();
rg.setContent(X_DD);
rg.getContentDataLocsMat();


'''
channel
'''
otfs.modulate(rg);

sigma2 = 1/np.power(10,SNR_d/10);

# his = np.asarray([[0.340993+0.124189j, -0.388739+0.410122j, -0.0250011+0.596597j,-0.846615+0.157308j,-0.0281749+0.0820766j,-0.304782-0.0582677j],[0.0373028-0.251298j,0.119883+0.481477j,-0.135965-0.772444j,0.0369038-0.32628j,-0.0339525+0.168356j,-0.24526-0.575117j]]);
# kis = np.asarray([[2,2,-3,-5,-3,0],[-3,-3,1,4,5,4]]);
# lis = np.asarray([[0,2,2,2,3,1],[3,2,2,0,3,2]]);

otfs.setChannel(p, lmax, kmax);
otfs.passChannel(No);
#otfs.passChannel(0);
his, lis, kis = otfs.getCSI();
H_DD = otfs.getChannel();

'''
Rx
'''
rg_rx = otfs.demodulate();
Y_DD = rg_rx.getContent();
yDD = np.reshape(Y_DD, [batch_size, M*N]);