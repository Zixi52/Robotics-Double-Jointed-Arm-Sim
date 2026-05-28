import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Arm params
L1 = 2.0          # length of upper arm
L2 = 1.5          # length of forearm
threshold = 0.05  # stop PID when end-effector error is below this

# Physics params
m1 = 1.0          # mass of upper arm (kg)
m2 = 0.75         # mass of forearm (kg)
g  = 9.81         # accel due to gravity
kd = 0.95         # inherent joint velocity damping (friction/motor resistanc, match animation interval
dt = 0.03         # animation timestep (s)
dt_phys = 0.005   # timestep for integrators

# PID gains: tune these to change arm behavior
# Kp: stiffness; higher = faster response but more overshoot
# Ki: integral; corrects persistent steady-state error (e.g. gravity sag)
# Kd: derivative; dampens oscillation
Kp = np.diag([180.0, 120.0])
Ki = np.diag([8, 8])
Kd = np.diag([45, 32])

# State
theta = np.array([np.pi / 4, np.pi / 4]) # [theta1, theta2]
theta_dot = np.array([0.0, 0.0])         # joint velocities
integral_error = np.array([0.0, 0.0])    # accumulated error for I term
theta_target = None                      # target joint angles (from IK)
target = None                            # target end-effector pos (from click)
theta_target_step = theta.copy()

# Kinematics
def forward_kinematics(theta):
    # Return (base, elbow, end_effector) as 2D points
    t1, t2 = theta
    base = np.array([0.0, 0.0])
    elbow = base + L1 * np.array([np.cos(t1), np.sin(t1)])
    end = elbow + L2 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])
    return base, elbow, end


def inertia_matrix(theta):
    t2 = theta[1]

    I1 = m1 * (L1**2) / 3
    I2 = m2 * (L2**2) / 3

    M11 = I1 + I2 + m2 * (L1**2 + 2 * L1 * (L2/2) * np.cos(t2))
    M12 = I2 + m2 * L1 * (L2/2) * np.cos(t2)
    M22 = I2

    return np.array([[M11, M12], [M12, M22]])


def coriolis_centrifugal(theta, theta_dot):
    t2 = theta[1]
    d1, d2 = theta_dot

    h = -0.5 * m2 * L1 * (L2/2) * np.sin(t2)

    return np.array([h * (2*d1*d2 + d2**2), h * (d1**2)])


def gravity_vector(theta):
    t1, t2 = theta

    g1 = (m1 * g * (L1/2) * np.cos(t1) +
          m2 * g * (L1*np.cos(t1) + (L2/2)*np.cos(t1 + t2)))

    g2 = m2 * g * (L2/2) * np.cos(t1 + t2)

    return np.array([g1, g2])


def pid_step(theta, theta_dot, theta_target, integral_error):
    e = theta_target - theta    # proportional error
    e = np.arctan2(np.sin(e), np.cos(e)) # angle wrap [-pi, pi]
    integral_error = integral_error + e * dt_phys    # accumulate integral
    e_dot = -theta_dot    # derivative of error

    tau = Kp @ e + Ki @ integral_error + Kd @ e_dot

    return tau, integral_error


fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-(L1 + L2 + 1), L1 + L2 + 1)
ax.set_ylim(-(L1 + L2 + 1), L1 + L2 + 1)
ax.set_aspect('equal')
ax.set_facecolor('#0f0f0f')
fig.patch.set_facecolor('#0f0f0f')
ax.grid(True, color='#333333', linewidth=0.5)
ax.set_title('2D Robotic Arm - click to set target', color='white', fontsize=12)
ax.tick_params(colors='#666666')

# Draw workspace boundaries (max and min reach circles)
outer = plt.Circle((0, 0), L1 + L2, color='#333333',
                        fill=False, linestyle='--', linewidth=1)
inner = plt.Circle((0, 0), L1 - L2, color='#333333',
                        fill=False, linestyle='--', linewidth=1)
ax.add_patch(outer)
ax.add_patch(inner)

# Arm segments
line_upper, = ax.plot([], [], color='#00aaff', linewidth=5,
                      solid_capstyle='round', label='Upper arm')
line_fore,  = ax.plot([], [], color='#00ffcc', linewidth=4,
                      solid_capstyle='round', label='Forearm')

# Joints
dot_base,  = ax.plot([], [], 'o', color='white', markersize=10)
dot_elbow, = ax.plot([], [], 'o', color='#aaaaaa', markersize=8)
dot_end,   = ax.plot([], [], 'o', color='#ffdd00', markersize=9)

# Target marker
dot_target, = ax.plot([], [], 'x', color='#ff4444', markersize=12)

info_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                    color='#aaaaaa', fontsize=9, va='top', fontfamily='monospace')

def update_display(theta, tau_pid):
    base, elbow, end = forward_kinematics(theta)

    line_upper.set_data([base[0], elbow[0]], [base[1], elbow[1]])
    line_fore.set_data([elbow[0], end[0]], [elbow[1], end[1]])

    dot_base.set_data([base[0]], [base[1]])
    dot_elbow.set_data([elbow[0]], [elbow[1]])
    dot_end.set_data([end[0]], [end[1]])

    if target is not None:
        dot_target.set_data([target[0]], [target[1]])
    else:
        dot_target.set_data([], [])

    err = np.linalg.norm(target - end) if target is not None else 0.0

    info_text.set_text(
        f"θ1 = {np.degrees(theta[0]):6.1f}°\n"
        f"θ2 = {np.degrees(theta[1]):6.1f}°\n"
        f"error = {err:.3f}\n"
        f"tau = [{tau_pid[0]:.2f}, {tau_pid[1]:.2f}]"
    )


def animate(_frame):
    global theta, theta_dot, theta_target, theta_target_step, target, integral_error

    tau_pid = np.array([0.0, 0.0])

    substeps = int(dt / dt_phys)

    for _ in range(substeps):

        if theta_target is not None:
            _, _, end = forward_kinematics(theta)

            dist = np.linalg.norm(target - end)

            if np.linalg.norm(dist) < threshold:
                integral_error = np.array([0.0, 0.0])

            theta_target_step += 0.1 * (theta_target - theta_target_step)

            tau_pid, integral_error = pid_step(
                theta, theta_dot, theta_target_step, integral_error
            )

            gain = np.clip(dist / 0.08, 0.0, 1.0)
            tau_pid *= gain

        M = inertia_matrix(theta)
        C = coriolis_centrifugal(theta, theta_dot)
        G = gravity_vector(theta)

        tau_damp = -10.0 * theta_dot

        if theta_target is None:
            tau = tau_damp
        else:
            tau = tau_pid + tau_damp + G
            tau = np.clip(tau, -100, 100)

        theta_ddot = np.linalg.solve(M, tau - C - G)

        theta_ddot = np.clip(theta_ddot, -50, 50)

        theta_dot += theta_ddot * dt_phys
        theta_dot = np.clip(theta_dot, -15, 15)

        theta += theta_dot * dt_phys

    update_display(theta, tau_pid)
    return line_upper, line_fore, dot_base, dot_elbow, dot_end, dot_target, info_text


def on_click(event):
    global target, theta_target, integral_error
    if event.inaxes != ax:
        return
    clicked = np.array([event.xdata, event.ydata])

    if L1 - L2 <= np.linalg.norm(clicked) <= L1 + L2:
        target = clicked
        theta_target = analytic_ik(target)      # solve for joint angles once on click
        integral_error = np.array([0.0, 0.0])   # reset integrator for new target
    else:
        print("Target outside workspace")


def analytic_ik(target):
    x, y = target

    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1, 1)

    t2 = np.arctan2(-np.sqrt(1 - D**2), D)
    t1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(t2), L1 + L2*np.cos(t2))

    return np.array([t1, t2])


fig.canvas.mpl_connect('button_press_event', on_click)

ani = animation.FuncAnimation(
    fig, animate, interval=30, blit=True, cache_frame_data=False
)

plt.tight_layout()
plt.show()