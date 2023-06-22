# question 1
import random as rand
import matplotlib.pyplot as plt
import numpy as np

# constants
e_m = 9.11e-31  # electron mass in Kg
e_q = -1.6e-19  # electron charge in C
K = 8.99e9  # colon force constant N*m^2*C^-2

tou = 1e-15  # sec
E0 = 30  # V/m
v0_initial = 2e-3  # m/sec


def electron_location_tou(x, y, v0, E1, E2, t):
    """This function give a new location of the particle based on time, current location, size of velocity v, the electric field and
    the size of the time step taken """
    vx = rand.choice((1, -1)) * rand.uniform(0.0, v0)  # randomize the velocity in x axi based on v0
    vy = rand.choice((1, -1)) * (v0 ** 2 - vx ** 2) ** 0.5  # calculate the velocity in the y axi based on v0 and
    # the velocity at the x axi
    e_speed_initial1 = vx
    e_speed_initial2 = vy
    force1 = e_q * E1  # Calculate the force in each axi based on the electric field
    force2 = e_q * E2
    e_speed_new1 = (force1 / e_m) * t + e_speed_initial1  # Calculate the new speed based on the force
    e_speed_new2 = (force2 / e_m) * t + e_speed_initial2
    e_loc_new1 = e_speed_new1 * t + x  # Calculate the new location
    e_loc_new2 = e_speed_new2 * t + y
    return e_loc_new1, e_loc_new2


def creat_xy_vectors(v0, num_of_points):
    """This function generate location and time vectors based on the number of time steps."""
    e_loc_new1, e_loc_new2 = electron_location_tou(0, 0, v0, E0, 0, tou)
    x_vector = [0, e_loc_new1]
    y_vector = [0, e_loc_new2]

    for point in range(1, num_of_points - 1):
        e_loc_new1, e_loc_new2 = electron_location_tou(e_loc_new1, e_loc_new2, v0, E0, 0, tou)
        x_vector += [e_loc_new1]
        y_vector += [e_loc_new2]
    created_t = [j * tou for j in range(num_of_points)]
    return x_vector, y_vector, created_t


x, y, t = creat_xy_vectors(v0_initial, 100)  # creation of the location and time vectors for q1

# סעיף 2
plt.figure()
plt.plot(t, x)
plt.title("x as function of t")
plt.ylabel("x[m]")
plt.xlabel("t[sec]")
fig2 = plt.figure()
plt.plot(t, y)
plt.title("y as function of t")
plt.ylabel("y[m]")
plt.xlabel("t[sec]")

z = t
plt.figure()
axis = plt.axes(projection='3d')
axis.scatter(x, y, z, c='r', marker='o')
axis.set_title("x and y as a function of t")
axis.set_xlabel('x[m]')
axis.set_ylabel('y[m]')
axis.set_zlabel('t[sec]')

plt.show()

# סעיף 3
x_end = x[-1]
t_end = t[-1]
v_drift = x_end / t_end  # Calculation of drift speed
print(f"The drift speed is {v_drift} m/s")
v_drift_n = []
for i in range(10000):  # Loop of 10k different drift speeds in order to sensitivity to change in size of speed changes
    v0_new = v0_initial + rand.choice((1, -1)) * rand.uniform(0.1, 1)
    creat_xy_vectors(v0_new, 100)
    x_end = x[-1]
    t_end = t[-1]
    v_drift_n += [x_end / t_end]
v_drift_avr = np.mean(v_drift_n)  # calculation of the mean of the 10k repetitions
v_drift_std = np.std(v_drift_n)  # calculation of the standard deviation of the 10k repetitions
print("average", v_drift_avr)
print("Standard Deviation", v_drift_std)
print(
    f"original drift speed {v_drift} m/s and average of the new drift speeds {v_drift_avr} m/s "
    f"with a standard deviation of {v_drift_std} m/s and the difference of {v_drift - v_drift_avr} m/s")

# סעיף 4
x, y, t = creat_xy_vectors(200, 100)  # creation of the location and time vectors for q4

plt.figure()
plt.plot(t, x)
plt.title("x as function of t")
plt.ylabel("x[m]")
plt.xlabel("t[sec]")
plt.figure()
plt.plot(t, y)
plt.title("y as function of t")
plt.ylabel("y[m]")
plt.xlabel("t[sec]")

plt.figure()
axis = plt.axes(projection='3d')
axis.scatter(x, y, t, c='r', marker='o')
axis.set_title("x and y as a function of t")
axis.set_xlabel('x[m]')
axis.set_ylabel('y[m]')
axis.set_zlabel('t[sec]')

plt.show()
