#-------------------------------------------------------------------------------
# Purpose:     Plots Trojans across random positions and velocities the result
#              is saved to a file. This file can then be plotted using this
#              program.

# BGN:	       6946S
# Created:     23/02/2021
#-------------------------------------------------------------------------------
import numpy as np
import csv, os

from Simulation import Simulation
from AsteroidGenerators import AstGen

FILE_NAME = '--.csv'

T_MAX = 800         # Total length of simulation / periods.
START_INTERVAL = 8  # Length of initial run to remove highly unstable asteroids.
T_INTERVAL = 132    # Time intervals to check wander distance after initial run.

N = 8               # Number of asteroids to simulate.

WD_LIM = 8          # Wander distance where an asteroid is not Trojan.

SPATIAL_LIM = 0.1   # Spatial size of anulus.
VEL_LIM = 0.1       # Maximum speed.


def calculate():

    # Container of the final wander distance of the asteroids.
    wander_distances = np.empty((0, 5))

    sim = Simulation(AstGen.small_displacements,
                     N,
                     SPATIAL_LIM,
                     VEL_LIM)

    # Run one short sim to remove all the immediately unstable asteroids.
    sim.run(START_INTERVAL)

    for times in range(START_INTERVAL, T_MAX+1, T_INTERVAL):

        intermediate_wander_distances = sim.wander_distance()

        for asteroid in intermediate_wander_distances:

            # Remove asteroids that are no longer Trojan.
            if asteroid[4] > WD_LIM:
                wander_distances = np.append(wander_distances,
                                             [asteroid],
                                             axis=0)
                sim.remove_asteroid(asteroid[:4])

        if len(sim.paths) == 0:
            break

        # Check progress of removing asteroids.
        print('After', times, 'periods', len(sim.paths), 'asteroids remain')

        # Dont run again - saves time.
        if times < T_MAX:
            sim.run(T_INTERVAL, True)

    # Add remaining asteroids.
    intermediate_wander_distances = sim.wander_distance()
    if len(intermediate_wander_distances) != 0:
        wander_distances = np.append(wander_distances,
                                     intermediate_wander_distances,
                                     axis=0)

    # Write data to file.
    with open(FILE_NAME, 'a', newline='') as writefile:
        writer = csv.writer(writefile, delimiter=',')

        # Write header with parameters of the simulation required for plotting.
        if os.path.getsize(FILE_NAME) == 0:
            writer.writerow([sim.R,
                             sim.r_sun,
                             sim.omega,
                             sim.L4[0],
                             sim.L4[1]])

        # Write wander distances.
        writer.writerows(wander_distances)


if __name__ == '__main__':
    calculate()
