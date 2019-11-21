import numpy as np
import matplotlib.pyplot as plt

def main():
    # Discretization parameters
    dx = 1./100.
    L = 1
    nx = int(L/dx)

    dt = 0.5*dx**2
    T = 1
    nt = int(T/dt)
    CX = dt/dx**2

    # Initialize arrays
    u = np.zeros(nx)
    u_sc = np.zeros(nx)
    t = np.linspace(0, T, nt)
    x = np.linspace(0, L, nx)

    # Initial values
    u0 = np.sin(np.pi*x)
    u0[0] = u0[-1] = 0
    u1 = u0             # u1 = solution at previous time step

    for j in range(nt):
        u[1:-1] = u1[1:-1] + CX*(u1[2:] - 2*u1[1:-1] + u1[0:-2]) # vectorized

        # for i in range(1,nx-1):
        #     u_sc[i] = u1[i] + CX*(u1[i+1] - 2*u1[i] + u1[i-1])   # scalar

        # print (np.mean(abs(u_sc - u)))
        # plt.plot(x, u0)
        # plt.plot(x, u, 'r--')
        # plt.axis([0,1,0,1])
        # plt.pause(0.01)
        # plt.clf()
        u1 = u

    plt.plot(x, u0)
    plt.plot(x, u, 'r--')

    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid('on')
    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()

main()
