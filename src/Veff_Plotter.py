#-------------------------------------------------------------------------------
# Purpose:     Calculates the effective potential to draw a graph of the
#              potential

# BGN:	       6946S
# Created:     23/03/2021
#-------------------------------------------------------------------------------
import numpy as np
from numpy import pi
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Physical Constants.
G = 4 * pi**2
M_SUN = 1
M_PLANET = 0.2
R = 5.2

# Plot Parameters.
X_RANGE = 15
Y_RANGE = 15
SPACE_RES = 200  # resolution of plot


def mag(r):
    """Calculates the magnitude of the r"""
    return (r[0]**2 + r[1]**2)**0.5


def main():
    rcParams['font.family'] = 'Palatino Linotype'
    fig, ax = plt.subplots(figsize=(5, 5))

    r_sun = -R * M_PLANET / (M_PLANET + M_SUN)
    r_planet = r_sun + R
    sun_pos = np.array([0, r_sun])
    planet_pos = np.array([0, r_planet])
    L_4 = [-R*np.sin(pi/3), R*np.cos(pi/3)+r_sun]
    L_5 = [R*np.sin(pi/3), R*np.cos(pi/3)+r_sun]

    xspace = np.linspace(-X_RANGE, X_RANGE, SPACE_RES)
    yspace = np.linspace(-Y_RANGE, Y_RANGE, SPACE_RES)
    [Xi, Yi] = np.meshgrid(xspace, yspace)
    space_grid = np.reshape(np.array([Xi, Yi]).T, (SPACE_RES**2, 2))

    centrifugal_coefficient = (M_SUN+M_PLANET)/(2*mag(sun_pos-planet_pos)**3)
    potential = -G*np.array([M_SUN/mag(r-sun_pos) +
                             M_PLANET/mag(r-planet_pos) +
                             centrifugal_coefficient*mag(r)**2
                             for r in space_grid])

    potential_grid = griddata((Xi.flatten(), Yi.flatten()), potential, (Xi, Yi))

    ax.contourf(Yi, Xi, potential_grid,
                np.linspace(1.75*max(potential),
                            max(potential),
                            15),
                cmap='gray')

    ax.plot(L_4[0], L_4[1], 'k+')
    ax.plot(L_5[0], L_5[1], 'k+')
    ax.plot(0, 0, '+', markersize='10', mew='1.5')

    ax.text(L_4[0]-0.15, L_4[1], 'L4', ha='right')
    ax.text(L_5[0]+0.15, L_5[1], 'L5')

    u = np.linspace(0, 2*np.pi, 100)
    ax.plot(r_planet*np.sin(u), r_planet*np.cos(u),
            '--', linewidth=0.6, color='black')
    ax.plot(r_sun*np.sin(u), r_sun*np.cos(u),
            '--', linewidth=0.6, color='black')

    jupiter = plt.Circle((0, r_planet), 0.2, color='black')
    sun = plt.Circle((0, r_sun), 0.4, color='black')
    ax.add_patch(jupiter)
    ax.add_patch(sun)

    ax.set_xlabel('x / au')
    ax.set_ylabel('y / au')
    ax.set_aspect('equal')
    plt.xlim(-7.4, 7.4)
    plt.ylim(-7.4, 7.4)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
