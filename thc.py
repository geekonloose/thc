import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import itertools
nodes = [1, 2, 3, 4, 5]    # total no of nodes
hs = [1]    # heat structure nodes
fn = [2, 3, 4, 5]    # fluid nodes
links = [1, 2, 3, 4, 5]
fl = [2, 3, 4, 5]
cond_links = []
conv_links = [1]
# T = np.ones(len(hs) + len(fn))
# Ts = np.ones(len(hs))
# Tf = np.ones(len(fn))
# sodium specific heat
cp = 1.3e3

# A_k/L_k matrix
a_l = np.diag(np.ones(len(fl)))

# mass flow rate matrix
amw = np.array([[-1, 0, 0, 0],
                [1, -1, 0, 0],
                [0, 1, -1, 0],
                [0, 0, 1, -1]])

# pressure difference matrix
awp = - a_l.dot(amw.transpose())

friction_matrix = np.diag(np.ones(len(fl))).dot(a_l)*0.2

ahhq = np.array([-1, 0, 0, 0, 0])
ahhq = ahhq.reshape(len(hs), len(links))
ahfq = np.array([[1, -1, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])
aqt = np.hstack([-ahhq.transpose(), -ahfq.transpose()])
u_matrix = np.diag(np.ones(aqt.shape[0])) 
q = u_matrix.dot(aqt)
ahfqt = ahfq.dot(q)
ahhqt = ahhq.dot(q)

ahfqt1 = ahfqt[:, :len(hs)]
ahfqt2 = ahfqt[:, len(hs):]

ahhqt1 = ahhqt[:, :len(hs)]
ahhqt2 = ahhqt[:, len(hs):]
dt = 0.01    # seconds

def mat_f(ahw):
    mat1 = np.hstack([np.zeros([amw.shape[0], awp.shape[1]]), amw,
                      np.zeros([amw.shape[0], ahhqt1.shape[1]]),
                      np.zeros([amw.shape[0], ahhqt2.shape[1]])])
    mat2 = np.hstack([awp, friction_matrix,
                      np.zeros([awp.shape[0], ahhqt1.shape[1]]),
                      np.zeros([amw.shape[0], ahhqt2.shape[1]])])

    mat3 = np.hstack([np.zeros([ahhqt1.shape[0], awp.shape[1]]),
                      np.zeros([ahhqt1.shape[0], amw.shape[1]]),
                      ahhqt1, ahhqt2])
    mat4 = np.hstack([np.zeros([ahw.shape[0], awp.shape[1]]),
                      ahw, ahfqt1, ahfqt2])
    # l = np.array([1, 1, 1, 2, 1])
    theta = np.radians([0, 0, 90, 180, 270])
    bm = np.ones([amw.shape[0]])
    bw = np.zeros([a_l.shape[0]]) 
    bw[1] = 5e5
    bhs = np.zeros([ahhqt1.shape[0]])
    bhf = np.zeros([ahw.shape[0]])

    b = np.hstack((bm, bw, bhs, bhf))
    b = b.copy()
    mat = np.vstack((mat1, mat2, mat3, mat4))
    return mat, b


initial_val = np.zeros(len(fl)+len(fl)+len(nodes))
# pressure initial value
initial_val[:len(fl)] = 1e5
initial_val[len(fl):2*len(fl)] = 10
# initial_val[len(fl):len(fl)+len(fl)] = 0.1
initial_val[len(fl)+len(fl):len(fl)+len(fl)+len(hs)] = 500 + 298
initial_val[len(fl)+len(fl)+len(hs):
            len(fl)+len(fl)+len(hs)+len(fn)] = 25+295
initial_val[-1] = 1000
initial_val[-2] = 600
initial_val[-3] = 700
initial_val[-4] = 400

x_old = initial_val.copy()


def f(x):
    spl_mat = np.ones_like(initial_val)
    spl_mat[:amw.shape[0]] = 0
    return spl_mat * x - spl_mat * x_old - mat.dot(x) * dt - b * dt


pressure = []
temp_hs = []
temp_fn = []

mass = np.ones(len(fl))*800
for i in range(10):
    temp = []
    temp.append(x_old[len(fl)*2:len(nodes)+2*len(fl)])
    # temp.append(temp[0])
    temp = list(itertools.chain.from_iterable(temp))

    if i == 0:
        dT = np.array([abs(temp[1]-temp[2]), abs(temp[2]-temp[3]),
                       abs(temp[3]-temp[4]), abs(temp[4]-temp[1])])
    else:
        dT = np.array([abs(temp[1]-temp[2]), abs(temp[2]-temp[3]),
                       abs(temp[3]-temp[4]), abs(temp[4]-temp[1])])
        mass = mass + amw.dot(w) * dt

    ahw = amw.dot(cp*np.diag(dT/mass))

    mat, b = mat_f(ahw)
    print('                  iteration no            ', i)
    print('x_old', x_old)
    print('mat', mat)
    print('b', b)
    print('ahw', ahw)
    print('dT', dT)
    print('mass', mass)
    x_old = opt.newton(f, x_old, maxiter=1000000)

    w = x_old[amw.shape[0]:amw.shape[0]+awp.shape[0]]

    pressure.append(x_old[:len(fl)])

    temp_hs.append(x_old[len(fl)+len(fl):len(fl)+len(nodes)+len(hs)])
    temp_fn.append(x_old[len(fl)+len(fl)+len(hs):2*len(fl)+len(nodes)])


plt.plot(pressure, label='pressure')
plt.plot(temp_hs, label='temp_hs')
# plt.plot(temp_fn, label='temp_fn')
plt.legend()
plt.show()
