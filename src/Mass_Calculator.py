#-------------------------------------------------------------------------------
# Purpose:     Calculate wander distances for different masses of planets
#              results are saved to a file.

# BGN:	       6946S
# Created:     01/04/2021
#-------------------------------------------------------------------------------
import numpy as np
import csv, os

from Simulation import Simulation
from AsteroidGenerators import AstGen

FILE_NAME = '--.csv'

SPATIAL_LIM = 1.2   # Spatial size of anulus.
VEL_LIM = 0.5       # Maximum speed of asteroids.

N_AST = 1000        # Number of asteroids to simulate for each mass.
WD_LIM = 8          # Wander distance where an asteroid is not Trojan.

T_MAX = 800         # Total length of simulation / periods.
START_INTERVAL = 10 # Length of initial run to remove highly unstable asteroids.
T_INTERVAL = 79     # Time intervals to check wander distance after initial run.

N_MASS = 45         # Number of masses to simulate.
MASSES = np.linspace(0.042, 0, N_MASS) # Masses to check.


def calculate():
    """
    Calculate the wander distances of asteroid for various masses and save
    the results to a file.
    """

    for mass in MASSES:

        print(mass)

        # Container of the wander distance of the asteroids
        wander_distances = np.empty((0,5))

        # Create simulation
        sim = Simulation(AstGen.uniform_r_correct_angular_velocity,
                         N_AST,
                         SPATIAL_LIM,
                         VEL_LIM,
                         planet_mass=mass)

        # Run one short sim to remove all the initially unstable asteroids
        sim.run(START_INTERVAL)

        for times in range(START_INTERVAL, T_MAX+1, T_INTERVAL):

            intermediate_wander_distances = sim.wander_distance()
            for asteroid in intermediate_wander_distances:

                # Remove asteroids that are no longer Trojan
                if asteroid[4] > WD_LIM:
                    wander_distances = np.append(wander_distances,
                                                 [asteroid],
                                                 axis=0)
                    sim.remove_asteroid(asteroid[:4])

            # If no asteroids left then leave the loop
            if len(sim.paths) == 0 :
                break

            # Check number of remaining stable asteroids.
            print('After', times, 'periods', len(sim.paths), 'asteroids remain')

            if times < T_MAX:
                sim.run(T_INTERVAL, True)

        # Add remaining asteroids
        intermediate_wander_distances = sim.wander_distance()

        if len(intermediate_wander_distances) != 0:
            wander_distances = np.append(wander_distances,
                                         intermediate_wander_distances,
                                         axis=0)


        # Write data to file
        with open(FILE_NAME, 'a', newline='') as writefile:
            writer = csv.writer(writefile, delimiter=',')

            # Write header only once per file.
            if os.path.getsize(FILE_NAME) == 0:
                writer.writerow([N_MASS, N_AST])

            # Write header of each mass for the specific simulation parameters.
            writer.writerow(['--', sim.R, sim.omega,
                             sim.L4[0], sim.L4[1], mass])

            writer.writerows(wander_distances)


if __name__ == '__main__':
    calculate()
