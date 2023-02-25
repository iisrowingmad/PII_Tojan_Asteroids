#-------------------------------------------------------------------------------
# Purpose:     Plots the angular position over time to calculate oscillation
#              period

# BGN:	       6946S
# Created:     23/02/2021
#-------------------------------------------------------------------------------
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import rcParams

from Simulation import Simulation
from AsteroidGenerators import AstGen

T_MAX = 100   # Length of time to run simulation for in periods of the planet.
N = 20        # Number of asteroids to simulate.
WD_LIM = 8    # Limit of what is considered Trojan.

# Limits for position and velocity .
SPATIAL_LIM = 1
VEL_LIM = 1


def sin_fit(t, A_1, A_2, w, b):
    """Fit function to calculate period of oscillation."""
    return (A_1 * np.sin(w * t + b) + A_2)


def main():
    rcParams['font.family'] = 'Palatino Linotype'
    rcParams['font.size'] = 10
    fig, ax = plt.subplots()

    sim = Simulation(AstGen.random_r_correct_angular_velocity,
                     N,
                     SPATIAL_LIM,
                     VEL_LIM)

    paths = sim.run(T_MAX)

    wander_distances = sim.wander_distance()

    omegas = np.empty(0)
    plotted = False  # Only want to plot one of the paths
    for n, asteroid in enumerate(wander_distances):

            # Remove asteroids that are no longer trojan
            if asteroid[4] < WD_LIM:

                theta = (np.arctan2(paths[n].rx, paths[n].ry) - sim.theta_L4)

                popt, pcov = curve_fit(sinfit,
                                       paths[n].t,
                                       theta,
                                       p0=[2, 1, 0.04, 0])  # Estimate fit

                omegas = np.append(omegas, popt[2])

                print(popt)  # Print fit parameters

                if not plotted:
                    ax.plot(paths[n].t, theta, '-')
                    ax.plot(paths[n].t, sinfit(paths[n].t,
                                               popt[0],
                                               popt[1],
                                               popt[2],
                                               popt[3]), '--')
                    plotted = True

    ax.set_xlabel('Time / y')
    ax.set_ylabel('Angular displacement from L4 / rad')
    ax.legend(['Simulated Asteroid', 'Sinusoidal Fit'])
    ax.set_xlim(0, 300)

    plt.show()

    print('Number of stable asteroids calculated:', len(omegas))
    print('Average:', np.mean(omegas))
    print('Std:', np.std(omegas))
    print('Range:', np.max(omegas)-np.min(omegas))


if __name__ == '__main__':
    main()
