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

rho = 850    # kg/m3
mu = 4.15e-4
Dh = np.ones(len(fl)) * 0.1   # hydraulic diameter of the pipes
cp = 1.3e3    # specific heat of the sodium
dt = 1e-8    # seconds
nit = 100
area = np.ones(len(fl)) * 0.00785
link_len = np.ones(len(fl)) * 1
# A_k/L_k matrix
a_l = np.diag(np.ones(len(fl))) * area

# mass flow rate matrix
amw = np.array([[-1, 0, 0, 0],
                [1, -1, 0, 0],
                [0, 1, -1, 0],
                [0, 0, 1, -1]])

# pressure difference matrix
awp = - amw.transpose().dot(a_l)

# old friction fasctor
# friction_matrix = np.diag(np.ones(len(fl))*32*4.15e-4 *
#                           1/800/0.00785/0.1**2).dot(a_l)

ahhq = np.array([-1, 0, 0, 0, 0])
ahhq = ahhq.reshape(len(hs), len(links))
ahfq = np.array([[1, -1, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])

aqt = np.hstack([-ahhq.transpose(), -ahfq.transpose()])
u_matrix = np.diag(np.ones(aqt.shape[0])) * 0.0001
q = u_matrix.dot(aqt)
ahfqt = ahfq.dot(q)
ahhqt = ahhq.dot(q)

ahfqt1 = ahfqt[:, :len(hs)]
ahfqt2 = ahfqt[:, len(hs):]

ahhqt1 = ahhqt[:, :len(hs)]
ahhqt2 = ahhqt[:, len(hs):]


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
    bm = np.ones([amw.shape[0]]) + 1e5
    bw = np.zeros([awp.shape[0]]) 
    # bw[1] = 1.5e5
    bhs = np.zeros([ahhqt1.shape[0]])
    bhf = np.zeros([ahw.shape[0]])

    b = np.hstack((bm, bw, bhs, bhf))
    b = b.copy()
    mat = np.vstack((mat1, mat2, mat3, mat4))
    return mat, b


initial_val = np.zeros(len(fl)+len(fl)+len(nodes))
# pressure initial value
initial_val[:len(fl)] = 1e5
initial_val[len(fl):2*len(fl)] = 0
# initial_val[len(fl):len(fl)+len(fl)] = 0.1
initial_val[len(fl)+len(fl):len(fl)+len(fl)+len(hs)] = 500 + 298
initial_val[len(fl)+len(fl)+len(hs):
            len(fl)+len(fl)+len(hs)+len(fn)] = 25+295
initial_val[-1] = 25+295
initial_val[-2] = 25+295
initial_val[-3] = 25+295
initial_val[-4] = 25+295

x_old = initial_val.copy()


def f(x):
    spl_mat = np.ones_like(initial_val)
    spl_mat[:amw.shape[0]] = 0
    return spl_mat * x - spl_mat * x_old - mat.dot(x_old) * dt - b * dt


def fric(j, v):
    eps = 1.0E-15
    if abs(v) < eps:
        Re = rho*eps*Dh[j]/mu
    else:
        Re = rho*abs(v)*Dh[j]/mu

    if Re < 2000.0:
        print('Laminar')
        fr = 64.0/Re
    else:
        print('Turbulent flow', Re)
        fr = 0.0032+0.221/Re**0.237
    return fr


pressure = []
temp_hs = []
temp_fn = []
w = np.zeros([nit, len(fl)])
mass = np.ones(len(fl))*10*1
friction_matrix = np.eye(len(fl))

for i in range(nit):
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
        mass = mass + amw.dot(w[i, :]) * dt

    # friction_matrix = fic_mat()
    for i1 in range(len(fl)):
        print(w[i1, i1])
        friction_matrix[i1, i1] = fric(i1, w[i1, i1]/rho/area[i1]) * \
            link_len[i1]/Dh[i1] * abs(w[i1, i1])

    ahw = amw.dot(cp*np.diag(dT/mass))

    mat, b = mat_f(ahw)
    print('                  iteration no            ', i)
    print('x_old', x_old)
    print('mat', mat)
    print('b', b)
    print('ahw', ahw)
    print('dT', dT)
    print('mass', mass)
    # if i = 0:

    x_old = opt.fsolve(f, x_old, fprime=None, full_output=0, col_deriv=0,
                       xtol=1e-08, maxfev=0, band=None, epsfcn=None,
                       factor=100, diag=None)
    # w = x_old[amw.shape[0]:amw.shape[0]+awp.shape[0]]
    w[i, :] = x_old[amw.shape[0]:amw.shape[0]+awp.shape[0]]

    pressure.append(x_old[:len(fl)][2])
    # flow_rate.append(x_old[len(fl)+1])
    temp_hs.append(x_old[len(fl)+len(fl):2*len(fl)+len(hs)])
    temp_fn.append(x_old[2*len(fl)+len(hs):2*len(fl)+len(nodes)][0])

plt.figure()
plt.plot(pressure, label='pressure')
plt.legend()
plt.figure()
plt.plot(temp_hs, label='temp_hs')
plt.plot(temp_fn, label='temp_fn')
plt.legend()
plt.show()
