#-------------------------------------------------------------------------------
# Purpose:     Plots Trojans across random positions and velocities the result
#               is saved to a file. This file can then be plotted using this
#               program.

# BGN:	       6946S
# Created:     23/02/2021
#-------------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import griddata
import csv

import matplotlib.pyplot as plt
from matplotlib import rcParams

FILE_NAME = '0.08_0.1_800.csv'

N = 8            # Number of asteroids to simulate.
WD_LIM = 8       # Wander distance where an asteroid is not Trojan.
GRAPH_RES = 70   # Resolution of the plot - used to average over asteroids.


def extract_data_from_file():
    """
    Extract the data from the file in FILE_NAME.

    Returns:
        (np.array(N, 5)): Array of the initial positions and corresponding
            wander distances of the asteroids saved in FILE_NAME.
        (np.array(5)): Array of the parameters that the simulation was run
            using. This is in the order [R, r_sun, omega, L4[0], L4[1]].

    """
    wander_distances = np.empty((0, 5))
    with open(FILE_NAME, newline='') as csvfile:
        for n, row in enumerate(csvfile):
            splitted = list(map(float, row.split(',')))

            if n == 0:
                parameters = splitted
            else:
                wander_distances = np.append(wander_distances,
                                             [np.array(splitted)],
                                             axis=0)
    return wander_distances, parameters


def plot():
    """
    Function to make contour plots of the relationships between the initial
    position, velocity and wander distance of an asteroid.

    This also creates a quiver plot to show the density of asteroids
    that were investigated.
    """
    rcParams['font.family'] = 'Palatino Linotype'

    wander_distances, parameters = extract_data_from_file()
    L_4 = [0, 0]  # Container for L4 vector
    R, r_s, omega, L_4[0], L_4[1] = parameters

    r_L4 = (L_4[0]**2 + L_4[1]**2)**0.5
    theta_L4 = np.arctan2(L_4[0], L_4[1])

    wD = wander_distances.T  # Transpose as is easer to plot
    print('Number of asteroids:', len(wD[0]))

    # Set up arrays for making meshes
    r = (wD[0]**2+wD[1]**2)**0.5 - r_L4
    theta = np.arctan2(wD[0], wD[1]) - theta_L4
    vr = (wD[0]*wD[2]+wD[1]*wD[3]) / (wD[0]**2+wD[1]**2)**0.5
    vtheta = (wD[0]*wD[3]-wD[2]*wD[1]) / (wD[0]**2+wD[1]**2)

    variables = [r, theta, vr, vtheta]
    variables_linspace = []
    grids = []
    # Create equally spaced arrays to be used to create meshes to plot with
    for x in range(4):
        variables_linspace.append(np.linspace(min(variables[x]),
                                              max(variables[x]),
                                              GRAPH_RES))

    # Create meshes to plot with
    for x in range(4):  # iterate through each of the variables
        for y in range(x+1, 4):
            [Xi, Yi] = np.meshgrid(variables_linspace[x], variables_linspace[y])
            grids.append(griddata((variables[x], variables[y]), wD[4], (Xi, Yi)))

    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(7, 7))

    axes_lab = ['Radial displacement / au',
                'Angular displacement / rad',
                'Radial velocity / au y^-1',
                'Angular velocity / rad y^-1']

    n = 0
    for i in range(3):  # Iterate horizontally through grid
        for j in range(3):  # Iterate vertically through grid
            if i > j:
                axes[j, i].axis('off')
            else:
                cs = axes[j, i].contourf(variables_linspace[i],
                                         variables_linspace[j+1],
                                         grids[n],
                                         np.linspace(0, WD_LIM, WD_LIM*4 + 1),
                                         extend='both')

                axes[j, i].set_ylim(variables_linspace[j+1].min(),
                                    variables_linspace[j+1].max())

                if i == 0 and j == 2:
                    # Plot line of constant angular velocity
                    r = np.linspace(min(variables_linspace[0]),
                                    max(variables_linspace[0]), 50) + r_L4
                    axes[j, i].plot(r-r_L4,
                                    omega * (r_L4**2 - r**2) / r**2,
                                    'r--',
                                    linewidth=0.5)
                if i == 0:
                    axes[j, i].set_ylabel(axes_lab[i+1])
                if j == 2:
                    axes[j, i].set_xlabel(axes_lab[j])

                n += 1

    # Add colour bar
    cbar = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(cs, cax=cbar)
    cbar.tick_params(labelsize=8)
    cbar.set_ylabel('Wander distance / au', rotation=270, labelpad=10)

    # Plot of the position and velocity (as an arrow)
    # this is used to observe the density and distribution of asteroids
    fig, ax = plt.subplots()
    ax.quiver(wD[0], wD[1], wD[2], wD[3])
    ax.plot(L_4[0], L_4[1], 'r+')

    plt.show()

if __name__ == '__main__':
    plot()
