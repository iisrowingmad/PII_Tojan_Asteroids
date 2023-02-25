#-------------------------------------------------------------------------------
# Purpose:     Basisic Plot of asteroids paths

# BGN:	       6946S
# Created:     23/02/2021
#-------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams

from Simulation import Simulation
from AsteroidGenerators import AstGen

T_MAX = 100    # Length of time to run simulation for in periods of the planet.
N = 1          # Number of asteroids to simulate.

# Limits for position and velocity (may not be required).
SPATIAL_LIM = 0.02
VEL_LIM = 0.02

WD_LIM = 8     # Maximum wander distance to classify as Trojan.
R_MARS  = 1.5  # Mars' radius.

def main():
    # Plot parametrs
    rcParams['font.family'] = 'Palatino Linotype'
    rcParams['font.size'] = 10
    fig, ax= plt.subplots(2, 1)

    sim = Simulation(AstGen.small_displacements,
                     N,
                     SPATIAL_LIM,
                     VEL_LIM)

    # Run with high presision to produce clean plots.
    paths = sim.run(12, min_steps_per_orbit=50)

    wander_distances = sim.wander_distance()

    # Often useful to change reference frame for plotting.
    #paths = sim.rotation_transformation(0)

    # Plot asteroids
    for n, asteroid_path in enumerate(paths):
        # Remove asteroids that are no longer trojan.
        if wander_distances[n][4] < WD_LIM:
            ax[0].plot(asteroid_path.rx, asteroid_path.ry, '-')

    # Plot orbit paths for Jupiter, Mars and L4.
    u = np.linspace(-np.pi, -np.pi, 500)
    #ax.plot(sim.r_planet*np.sin(u), sim.r_planet*np.cos(u),
            #'--', linewidth=0.5, color='darkorange')
    ax.plot(R_MARS*np.sin(u), R_MARS*np.cos(u)+sim.r_sun,
            '--', linewidth=0.5, color='red')
    ax[0].plot(sim.r_L4*np.sin(u), sim.r_L4*np.cos(u),
            '--', linewidth=0.5, color='black')

    # Add circles for bodies.
    jupiter = plt.Circle((0, sim.r_planet), 0.2, color='orange')
    sun = plt.Circle((0, sim.r_sun), 0.4, color='gold')
    ax.add_patch(jupiter)
    ax.add_patch(sun)

    # Add text for bodies.
    ax.text(0.4, R_MARS + sim.r_sun, "Mars' Orbit")
    ax.text(0.4, sim.r_planet, 'Jupiter')
    ax.text(0.4, sim.r_sun - 0.5, 'Sun')

    #ax.plot(0, 0, 'k+')
    ax.plot(sim.L4[0], sim.L4[1], 'k+')
    ax.plot(sim.L5[0], sim.L5[1], 'k+')
    ax.text(sim.L4[0]*(1.03), sim.L4[1], 'L4', ha='right')
    ax.text(sim.L5[0]*(1.03), sim.L5[1], 'L5')

    #ax.set_xlim(-5.24,-3.6)
    #ax.set_ylim(0.8, 3.8)
    ax.set_xlabel('x / au')
    ax.set_ylabel('y / au')
    ax.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    main()
