#This SEIR model was modified from  the python version of https://cs.uwaterloo.ca/~paforsyt/SEIR.html

import numpy as np
import parameters as parameters
from scipy import integrate
from calculations import seir_function
import matplotlib.pyplot as plt
import pandas as pd
from numpy import savetxt

S_0 = 328239522 #USA population excluding initial infected, exposed population,

I_0 =  1

E_0 = 1.0 * I_0  # initial exposed.

R_0 = 0 # none are recovered yet as of Jan 21.

N = S_0 + I_0 + E_0 + R_0  # N = total population

sigma = 1/5.2 #(average duration of incubation is 1/δ)
#transmission rate from exposed -> infected
#incubation rate
gamma = 1./5.2


c = 0 #mutations. not yet
"""
 R_zero = number of people infected by each infectious person
          this has nothing to do with "R" = removed above
          or R_0 (initial value of recovered)
          but is common terminology (confusing, but usual notation)
     time dependent, starts off large, then drops with"""
        # time due to public health actions (i.e. quarantine, social distancing)
""" R_zero > 1, cases increase
    R_zero < 1 cases peak and then drop off 
      R_zero declining with time https://www.nature.com/articles/s41421-020-0148-0
      beta = R_zero*gammma (done in "seir.m" )

     table of:   time(days)  R_zero
                  ....     ....
                  ....     ....
                  ....     ....
       linearly interpolate between times
       Note: this is different from Wang et al (2020), which assumes
             piecewise constant values for R_zero
"""
r_zero_array = np.zeros([8, 2])
r_zero_array[0, :] = [0.0, 2.2]  # t=0 days https://www.scientificamerican.com/article/how-does-the-new-coronavirus-compare-with-the-flu/
r_zero_array[2, :] = [30, 2.2] # 60 2/20
#For the following days this source was used: http://metrics.covid19-analysis.org/
r_zero_array[3, :] = [60.0, 2.19]  #  3/21
r_zero_array[4, :] = [70.0, 1.43]  #  3/31
r_zero_array[5, :] = [90.0, 1.04]  # 4/20
r_zero_array[6, :] = [120, 0.96]  # t 5/20
r_zero_array[7, :] = [150.0, 1.21 ] #6/19


#it's different for each state
params = parameters.Params(c, N, sigma, gamma, r_zero_array)
outputs = []
t_0 = 0
tspan = np.linspace(t_0, 184, 183)  # start to end of time in days, array
# time, size, index

y_init = np.zeros(4)
y_init[0] = S_0
y_init[1] = E_0
y_init[2] = I_0
y_init[3] = R_0


def seir_with_params(t, y):
    return seir_function(t, y, params)

r = integrate.ode(seir_with_params).set_integrator("dopri5")
r.set_initial_value(y_init, t_0)
y = np.zeros((len(tspan), len(y_init)))

y[0, :] = y_init  # array for solution
for i in range(1, 183):
    y[i, :] = r.integrate(tspan[i])
    if not r.successful():
        raise RuntimeError("Could not integrate")
# generate model
fig, axes = plt.subplots(ncols=4)
axes[0].plot(tspan, y[:, 0], color="b", label="S")
axes[1].plot(tspan, y[:, 1], color="r", label="E")
axes[0].set(xlabel="time (days)", ylabel="S: susceptible")
axes[1].set(xlabel="time (days)", ylabel="E: exposed")
axes[2].plot(tspan, y[:, 2], color="g", label="I: infectious")
axes[3].plot(tspan, y[:, 3], color="b", label="R: recovered")
axes[2].set(xlabel="time (days)", ylabel="I: infectious")
axes[3].set(xlabel="time (days)", ylabel="R: recovered")
axes[0].legend()
axes[1].legend()
axes[2].legend()
axes[3].legend()

plt.savefig('plot.png')
plt.show()

total_cases = y[:, 1] + y[:, 2] + y[:, 3]
total_cases_active = y[:, 1] + y[:, 2]

fig, ax = plt.subplots()
ax.plot(tspan, total_cases, color="b", label="E+I+R: Total cases")
ax.plot(tspan, total_cases_active, color="r", label="E+I: Active cases")
ax.set(xlabel="time (days)", ylabel="Patients", title='Cumulative and active cases')
plt.legend()
plt.savefig('total_cases.jpg')
plt.show()

nsteps = np.size(tspan)
S_end = y[nsteps - 1, 0]
E_end = y[nsteps - 1, 1]
I_end = y[nsteps - 1, 2]
R_end = y[nsteps - 1, 3]

total = S_end + E_end + I_end + R_end

print('time (days): % 2d' % tspan[nsteps - 1])

print('total population: % 2d' % total)

print('initial infected: % 2d' % I_0)

print('total cases (E+I+R) at t= % 2d : % 2d' % (tspan[nsteps - 1], E_end + I_end + R_end))

print('Recovered at t=  % 2d : % 2d \n' % (tspan[nsteps - 1], R_end))
print('Infected (infectious) at t= % 2d : % 2d \n' % (tspan[nsteps - 1], I_end))
print('Exposed (non-infectious) at t= % 2d : % 2d \n ' % (tspan[nsteps - 1], E_end))
print('Susceptable at t= % 2d : % 2d \n ' % (tspan[nsteps - 1], S_end))

#saved E,I,R cases in a csv file.
save= np.savetxt('outputs.csv', total_cases, delimiter = ',')
file = open('outputs.csv', 'r')
print(file.read())
