#-------------------------------------------------------------------------------
# Purpose:     Calculate how the Hamiltonian of the simulation varies over time
#              to check the simulation is accurate and stable.

# BGN:	       6946S
# Created:     23/02/2021
#-------------------------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from time import time

from Simulation import Simulation
from AsteroidGenerators import AstGen

T_MAX = 800     # Length of simulation in periods of the planet.
N = 1           # Number of asteroids to simulate.
MIN_STEPS = 12  # Min steps per orbit to be used in ODE solver.


def main():
    # Create Figure
    rcParams['font.family'] = 'Palatino Linotype'  # make font match report
    rcParams['font.size'] = 12
    fig, (ax1, ax2) = plt.subplots(1, 2)

    sim = Simulation(AstGen.hamiltonian_generated, N)

    # Run simulation
    start_time = time()
    paths = sim.run(T_MAX, min_steps_per_orbit=MIN_STEPS)
    print('Time for', MIN_STEPS, ':', time()-start_time)

    # Print wander distances of asteroids
    wanderdis = sim.wander_distance()
    for ast in wanderdis:
        print(ast[4])

    # Calculate and plot Hamiltonian
    for n_ast in range(len(paths)):
        t, H = sim.hamiltonian(sim.initial_positions[n_ast])
        ax2.plot(t, abs((H-H[0]) / H[0]) * 100, '-')

    for asteroid in paths:
        ax1.plot(asteroid.rx, asteroid.ry, '-')

    # Plot orbit paths L4.
    u = np.linspace(0, 2*np.pi, 500)
    ax1.plot(sim.r_L4*np.sin(u), sim.r_L4*np.cos(u),
             '--', linewidth=0.5, color='black')

    # Add circles for Sun/planet
    jupiter = plt.Circle((0, sim.r_planet), 0.2, color='orange')
    sun = plt.Circle((0, sim.r_sun), 0.4, color='gold')
    ax1.add_patch(jupiter)
    ax1.add_patch(sun)

    # Add text for Sun/planet.
    ax1.text(0.4, sim.r_planet, 'planet')
    ax1.text(0.4, sim.r_sun - 0.5, 'Sun')

    ax1.plot(0, 0, 'k+')
    ax1.plot(sim.L4[0], sim.L4[1], 'k+')
    ax1.text(sim.L4[0]*(1.03), sim.L4[1], 'L4', ha='right')

    ax2.set_xlabel('Time / y')
    ax2.set_ylabel('% difference from the initial Hamiltonian')
    ax2.set_box_aspect(1)

    ax1.set_xlabel('x / au')
    ax1.set_ylabel('y / au')
    ax1.set_aspect('equal')
    ax1.set_box_aspect(1)

    plt.tight_layout(h_pad=1)
    plt.show()

if __name__ == '__main__':
    main()
