#-------------------------------------------------------------------------------
# Purpose:     Collection of functions to create asteroids with different
#              distributions.

# BGN:	       6946S
# Created:     22/03/2021
#-------------------------------------------------------------------------------
import numpy as np
from random import uniform
from numpy import pi

from Simulation import Simulation


class AstGen:
    """
    A container for functions that generate asteroids in different
    patterns/distributions. All functions create asteroids centred on L4.

    All have the calling signature (N, sim, *args) where N is the number of
    asteroids to generate, sim is the instance of the simulation class that
    call the generating function - This is so the generating function can
    extract information such as the position of the Lagrange point or angular
    velocity. In general the additional arguments are related to the ranges of
    permitted velocities and space for a particular instance.

    The functions are named with the distribution of space first followed by
    the velocity distribution. So uniform_square_random_velocity would
    refer to asteroids placed uniformly in a square with each asteroid
    having a random velocity.

    The velocities are in the rotating frame of the simulation class and are
    therefore only uniform or random in the rotating frame of the simulation.

    """
    def L4_zero_velocity(N, sim):
        """
        Generate N asteroids located at L4 with no initial velocity in the
        rotating frame.
        """
        return np.array([np.array([sim.L4[0], sim.L4[1], 0, 0])]*N)

    def L4_random_velocity(N, sim, vel_lim):
        """
        Generate N asteroids at L4 with random velocities up to a maximum
        speed of vel_lim.
        """
        asteroids = np.zeros((N, 4))  # for x, y, vx, vy

        for ast in range(N):

            speed = uniform(0, vel_lim)
            vel_angle = uniform(0, 2*np.pi)

            asteroids[ast] = [sim.L4[0],
                              sim.L4[1],
                              speed*np.cos(vel_angle),
                              speed*np.sin(vel_angle)]

        return asteroids

    def uniform_annulus_sector_zero_velocity(N, sim, spatial_lim):
        """
        Approximately even distribution of asteroids across an annulus sector
        of side length 2*spatial_lim and angle 2*spatial_lim centred on L4.
        All asteroids are initially stationary within the simulation.
        """
        # approximate even distribution of asteroids in this anulus
        n = int((N / R)**0.5)

        thetas = np.linspace(sim.theta_L4 - spatial_lim,
                             sim.theta_L4 + spatial_lim,
                             int(N/n))

        rs = np.linspace(-spatial_lim, spatial_lim, n) + sim.r_L4

        asteroids = np.empty((0, 4))  # For [x, y, vx, vy].

        for theta in thetas:
            for r in rs:
                asteroids = np.append(asteroids,
                                      [[r*np.sin(theta),
                                        r*np.cos(theta),
                                        0,
                                        0]],
                                      axis=0)
        return asteroids

    def random_annulus_sector_random_velocity(N, sim, spatial_lim, vel_lim):
        """
        Random distribution of asteroids across an annulus sector
        of side length 2*spatial_lim and angle 2*spatial_lim centred on L4.
        Asteroids are given a random velocity with maximum magnitude vel_lim.
        """
        asteroids = np.zeros((N, 4))

        for asteroid in range(N):

            # Initially calculate wrt the sun then add offset as barycentre
            # is at the origin
            r = uniform(-spatial_lim, spatial_lim) + sim.r_L4
            theta = uniform(-spatial_lim, spatial_lim) + sim.theta_L4
            speed = uniform(0, vel_lim)
            vel_angle = uniform(0, 2*np.pi)

            asteroids[asteroid] = np.array([r*np.sin(theta),
                                            r*np.cos(theta),
                                            speed*np.sin(vel_angle),
                                            speed*np.cos(vel_angle)])

        return asteroids

    def random_r_correct_angular_velocity(N, sim, r_lim, vel_lim):
        """
        Creates a random distribution of asteroids across a radial range of
        -0.8*r_lim to r_lim centred on L4, with radial velocity range -vel_lim
        to vel_lim. Asteroids are placed at the same angle as L4 and their
        angular velocity is such that the angular momentum of the asteroid
        matches that of	a mass stationary at the Lagrange point L4.
        """
        asteroids = np.zeros((N, 4))

        for ast in range(N):
            '''
            The 0.8 included here is to limit the inward r as the plots
            are asymmetric in the r axis.
            '''
            r = uniform(-r_lim*(0.8), r_lim) + sim.r_L4
            theta = sim.theta_L4
            v_r = uniform(-vel_lim, vel_lim)
            v_theta = sim.omega * (sim.r_L4**2 - r**2) / r

            asteroids[ast] = [r*np.sin(theta),
                              r*np.cos(theta),
                              v_r*np.sin(theta) - v_theta*np.cos(theta),
                              v_r*np.cos(theta) + v_theta*np.sin(theta)]

        return asteroids

    def uniform_r_correct_angular_velocity(N, sim, r_lim, vel_lim):
        """
        Creates uniform distribution of asteroids across a radial range of
        2*r_lim centred on L4 and radial velocity range 2*vel_lim centred on 0.
        Asteroids are placed at the same angle as L4 and their angular velocity
        is such that the angular momentum of the asteroid matches that of the
        Lagrange point l4.
        """
        asteroids = np.zeros((0, 4))

        # Attempt to find integer ratio of No. velocities to No. radii such
        # that this ratio matches the ratio in the limits inputted.
        # (this is crude but works sufficiently well)
        ratio = (1.8*r_lim) / (2*vel_lim)
        n_v = int((N / ratio)**0.5)

        while N % n_v != 0:
            n_v -= 1
            if n_v == 1:
                raise Exception('failed')  # Just in case.

        rs = np.linspace(-r_lim*(0.8), r_lim, N // n_v) + sim.r_L4
        v_rs = np.linspace(-vel_lim, vel_lim, n_v)
        theta = sim.theta_L4

        for r in rs:
            for v_r in v_rs:

                v_theta = sim.omega * (sim.r_L4**2 - r**2) / r

                asteroids = np.append(asteroids,
                                      [[r*np.sin(theta),
                                        r*np.cos(theta),
                                        v_r*np.sin(theta)-v_theta*np.cos(theta),
                                        v_r*np.cos(theta)+v_theta*np.sin(theta)]],
                                      axis=0)

        return asteroids

    def small_displacements(N, sim, spatial_lim, vel_lim):
        """
        Creates asteroids with small spatial and velocity offsets from L4.
        8 asteroids are generated 2 for each of the r, vr, theta and v_thera
        directions with magnitudes specified by the spatial_lim and vel_lim
        parameters.
        """
        if N > 8:
            N = 8

        asteroids = np.zeros((N, 4))

        rs = [-spatial_lim, spatial_lim, 0, 0, 0, 0, 0, 0] + sim.r_L4
        v_rs = [0, 0, -vel_lim, vel_lim, 0, 0, 0, 0]
        thetas = [0, 0, 0, 0, -spatial_lim/sim.r_L4,
                  spatial_lim/sim.r_L4, 0, 0] + sim.theta_L4
        v_thetas = [0, 0, 0, 0, 0, 0, -vel_lim, vel_lim]

        for asteroid in range(N):

            r = rs[asteroid]
            v_r = v_rs[asteroid]
            theta = thetas[asteroid]
            v_theta = v_thetas[asteroid]

            asteroids[asteroid] = [r*np.sin(theta),
                                   r*np.cos(theta),
                                   v_r*np.sin(theta) - v_theta*np.cos(theta),
                                   v_r*np.cos(theta) + v_theta*np.sin(theta)]
        return asteroids

    def hamiltonian_generated(N, sim):
        """
        Five random positions and velocities calculated initially using a
        random generator but saved here so that different algorithms/tests can
        be compared on identical datasets.
        """
        return [[-4.5033321,  2.59480519, -0.05120358, -0.0408954],
                [-4.5033321,  2.59480519, -0.05614151, -0.02514371],
                [-4.5033321,  2.59480519,  0.01003798, -0.05837857],
                [-4.5033321,  2.59480519, -0.03436298, -0.02357075],
                [-4.5033321,  2.59480519,  0.00342264, -0.01532051]]
