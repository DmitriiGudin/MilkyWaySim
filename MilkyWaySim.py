from __future__ import division
import numpy as np
from numpy import genfromtxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab
import math
import csv
import time
import pdb
import sys
import random
import lib_distrib as l_d
import lib_gendistrib as l_gd
import pymp

from amuse import datamodel
from amuse.lab import *
from amuse.units import units, generic_unit_converter, constants, nbody_system, generic_unit_system
from amuse.datamodel import Particles, particle_attributes, ParticlesWithUnitsConverted
from amuse.ic.kingmodel import new_king_model
from amuse.ic.brokenimf import new_kroupa_mass_distribution
from amuse.community.gadget2.interface import Gadget2
from amuse.support import io

from pylab import *
from fractions import Fraction

def Print(text, start_time):
    print time.time()-start_time, "s: ", text

def Plots(dm_particles, gas_particles, t):
    star_particles = dm_particles.select(lambda m: not m==dm_particles[0].mass, ["mass"])
    DM_particles = dm_particles.select(lambda m: m==dm_particles[0].mass, ["mass"])
    coord_temp = ["y","z"]
    lims = [60, 120, 200]
    figname_temp = ["star", "gas", "DM"]
    colors = ["blue", "black", "red"]
    for i in range (0,2):
        for j in range (0,3):
	    plt.clf()
    	    eval("plt.scatter("+figname_temp[j]+"_particles.x.value_in(units.kpc), "+figname_temp[j]+"_particles."+coord_temp[i]+".value_in(units.kpc), marker='*', color='" + str(colors[j])+"')")
    	    plt.xlim (-lims[j],lims[j])
    	    plt.ylim (-lims[j],lims[j])
            plt.xlabel ("x (kpc)")
            plt.ylabel (str(coord_temp[i])+" (kpc)")
            plt.title ("Galaxy, " + str(figname_temp[j]) + " particles")
    	    plt.gcf().set_size_inches(12, 12)
    	    plt.gcf().savefig("plots/"+str(figname_temp[j])+"_x"+str(coord_temp[i])+"_"+str(t)+".png", dpi=100)

def gen_particles_Einasto(N, mass, N_CORES=1, r_limit=100):
    particles = Particles(N)
    def func(r): 
        return l_d.Einasto (r, R=50, a=0.5)
    r_distrib = l_gd.produce_distr(func, (0,), (r_limit,), N, N_CORES=N_CORES, lim_N_samples=1e5)
    radii = []
    for i in r_distrib: 
        radii.append(i[0])
    for r, p in zip (radii, particles):
        coord_xyz=l_gd.coord_from_r(r,3)
        p.x, p.y, p.z = coord_xyz[0] | units.kpc, coord_xyz[1] | units.kpc, coord_xyz[2] | units.kpc
        p.mass = mass/N | units.MSun
    return particles  

def gen_particles_Jaffe(N, mass, N_CORES=1, r_limit=3):
    particles = Particles(N)
    def func(r): 
        return l_d.Jaffe (r, R=15)
    r_distrib = l_gd.produce_distr(func, (0,), (r_limit,), N, N_CORES=N_CORES, lim_N_samples=1e5)
    radii = []
    for i in r_distrib: 
        radii.append(i[0])
    for r, p in zip (radii, particles):
        coord_xyz=l_gd.coord_from_r(r,3)
        p.x, p.y, p.z = coord_xyz[0] | units.kpc, coord_xyz[1] | units.kpc, coord_xyz[2] | units.kpc
        p.mass = mass/N | units.MSun
    return particles

def gen_particles_Hernquist(N, mass, N_CORES=1, r_limit=3):
    particles = Particles(N)
    def func(r): 
        return l_d.Hernquist (r, R=30)
    r_distrib = l_gd.produce_distr(func, (0,), (r_limit,), N, N_CORES=N_CORES, lim_N_samples=1e5)
    radii = []
    for i in r_distrib: 
        radii.append(i[0])
    for r, p in zip (radii, particles):
        coord_xyz=l_gd.coord_from_r(r,3)
        p.x, p.y, p.z = coord_xyz[0] | units.kpc, coord_xyz[1] | units.kpc, coord_xyz[2] | units.kpc
        p.mass = mass/N | units.MSun
    return particles

def gen_particles_exp_sech(N, mass, N_CORES=1, R_limit=30, z_limit=10):
    particles = Particles(N)
    def func(r, z): 
        return l_d.exp_sech(r, z, R=10, Z=1)
    rz_distrib = l_gd.produce_distr(func, (0,-z_limit), (R_limit,z_limit), N, N_CORES=N_CORES, lim_N_samples=1e5)
    radii, heights = [], []
    for i in rz_distrib: 
        radii.append(i[0])
        heights.append(i[1])
    for r, z, p in zip (radii, heights, particles):
        coord_xy=l_gd.coord_from_r(r,2)
        p.x, p.y, p.z = coord_xy[0] | units.kpc, coord_xy[1] | units.kpc, z | units.kpc
        p.mass = mass/N | units.MSun
    return particles  

def Gadget_config(SE):
    SE.parameters.epsilon_squared = 0.0001 | units.kpc**2
    SE.parameters.gas_epsilon = 0.01 | units.kpc
    SE.parameters.time_max = 1 | units.Gyr
    SE.parameters.max_size_timestep = 0.001 | units.Myr
    SE.parameters.min_size_timestep = 0.001*0.01 | units.Myr
    SE.parameters.time_limit_cpu = 1 | units.yr
    SE.parameters.stopping_condition_minimum_density = 0 | units.m**(-3) * units.kg
    SE.parameters.stopping_condition_minimum_internal_energy = -1e20 | units.m**2 * units.s**(-2)    

def simple_vphi_distr(all_particles, r_max): # To parallelize later
    velocity_list=[] 
    for i in np.arange(0,r_max):
        mass=0 | units.MSun
        for p in all_particles:
            if p.position.length()<i+1 | units.kpc: mass+=p.mass
        velocity_squared = (6.67408*1e-11 | units.m**3 / units.kg / units.s**2) * mass / (i+1 | units.kpc)
        velocity_list.append (math.sqrt(velocity_squared.value_in(units.km**2/units.s**2)) | units.km/units.s)
    return velocity_list

def assign_star_velocities(particles, all_particles, r_max): # To parallelize later
    velocity_list = simple_vphi_distr(all_particles, r_max)
    for p in particles:
        radius = p.position.length().value_in(units.kpc)
        fixed_radius = int(math.ceil(radius))    
        velocity = velocity_list[fixed_radius-1]
        radial_vector = (p.x.value_in(units.kpc) / (math.sqrt(p.x.value_in(units.kpc)**2 + p.y.value_in(units.kpc)**2)) , p.y.value_in(units.kpc) / (math.sqrt(p.x.value_in(units.kpc)**2 + p.y.value_in(units.kpc)**2)))
        p.vx = velocity*radial_vector[1]
        p.vy = -velocity*radial_vector[0]
        p.vz = 0 | units.m/units.s        

def assign_halo_velocities(particles): # To parallelize later
    Pot = abs(particles.potential_energy())
    N = len(particles)
    for p in particles:
        velocity_squared = Pot/(N*p.mass)
        v = math.sqrt(velocity_squared.value_in(units.m**2/units.s**2))
        v_directed = l_gd.coord_from_r(v,3)
        p.vx, p.vy, p.vz = v_directed[0] | units.m/units.s, v_directed[1] | units.m/units.s, v_directed[2] | units.m/units.s 

def Run(N_CORES, DM_halo_r_limit, star_disk_R_limit, star_disk_z_limit, star_bulge_r_limit):
    start_time = time.time()
    Print ("Generating dark matter halo...", start_time)
    DM_halo_particles = gen_particles_Einasto(N=10**2, mass=2*10**12, N_CORES=N_CORES, r_limit=DM_halo_r_limit)
    Print ("Generating disk stars...", start_time)
    star_disk_particles = gen_particles_exp_sech(N=10**2, mass=4*10**10, N_CORES=N_CORES, R_limit=star_disk_R_limit, z_limit=star_disk_z_limit)
    Print ("Generating bulge stars...", start_time)
    star_bulge_particles = gen_particles_Jaffe(N=10**1, mass=2*10**10, N_CORES=N_CORES, r_limit=star_bulge_r_limit)
    Print ("Generating disc gas...", start_time)
    gas_disk_particles = gen_particles_exp_sech(N=10**2, mass=1*10**10, N_CORES=N_CORES, R_limit=star_disk_R_limit, z_limit=star_disk_z_limit)
    Print ("Generating halo gas...", start_time)
    gas_halo_particles = gen_particles_Einasto(N=10**2, mass=1*10**10, N_CORES=N_CORES, r_limit=star_disk_R_limit*2)
    for p in datamodel.ParticlesSuperset([gas_disk_particles.as_set(),gas_halo_particles.as_set()]): p.u = 1 | units.km**2 / units.s**2
    Print ("Assigning velocities to non-halo particles...", start_time)
    assign_star_velocities(datamodel.ParticlesSuperset([star_disk_particles,star_bulge_particles,gas_disk_particles, gas_halo_particles]), datamodel.ParticlesSuperset([star_disk_particles,star_bulge_particles,gas_disk_particles,gas_halo_particles,DM_halo_particles]), 2*math.sqrt(star_disk_R_limit**2+star_disk_z_limit**2))
    Print ("Assigning velocities to halo particles...", start_time)
    assign_halo_velocities(datamodel.ParticlesSuperset([DM_halo_particles,gas_halo_particles]))
    Print ("Initializing Gadget2 code...", start_time)
    SE = Gadget2(number_of_workers=N_CORES-1)
    Gadget_config(SE)
    Print ("Adding particles to the code...", start_time)
    SE.dm_particles.add_particles(DM_halo_particles)
    SE.dm_particles.add_particles(star_disk_particles)
    SE.dm_particles.add_particles(star_bulge_particles)
    SE.gas_particles.add_particles(gas_disk_particles)
    SE.gas_particles.add_particles(gas_halo_particles)
    #io.write_set_to_file(SE.particles, 'output.dat', 'gadget')
    Print ("Starting the simulation...", start_time)
    for t in np.arange(0, 1000, 1):
        SE.evolve_model(t | units.Myr)
        Plots(SE.dm_particles, SE.gas_particles, t)
        Print ("Timestep "+str(t+1)+", screenshots saved!", start_time)

if __name__ == '__main__':
    N_CORES = 4
    DM_halo_r_limit = 100
    star_disk_R_limit = 30
    star_disk_z_limit = 10
    star_bulge_r_limit = 3
    Run(N_CORES, DM_halo_r_limit, star_disk_R_limit, star_disk_z_limit, star_bulge_r_limit)
