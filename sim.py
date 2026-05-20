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
kd = 0.95         # inherent joint velocity damping (friction/motor resistance)
dt = 0.03         # timestep (s), match animation interval

# PID gains: tune these to change arm behavior
# Kp: stiffness; higher = faster response but more overshoot
# Ki: integral; corrects persistent steady-state error (e.g. gravity sag)
# Kd: derivative; dampens oscillation
Kp = np.diag([175.0, 125.0])
Ki = np.diag([12, 10])
Kd = np.diag([20, 12])

# State
theta = np.array([np.pi / 4, np.pi / 4]) # [theta1, theta2]
theta_dot = np.array([0.0, 0.0])         # joint velocities
integral_error = np.array([0.0, 0.0])    # accumulated error for I term
theta_target = None                      # target joint angles (from IK)
target = None                            # target end-effector pos (from click)

# Kinematics
def forward_kinematics(theta):
    # Return (base, elbow, end_effector) as 2D points
    t1, t2 = theta
    base = np.array([0.0, 0.0])
    elbow = base + L1 * np.array([np.cos(t1), np.sin(t1)])
    end = elbow + L2 * np.array([np.cos(t1 + t2), np.sin(t1 + t2)])
    return base, elbow, end

def jacobian(theta):
    # 2x2 Jacobian matrix J where [dx, dy]^T = J @ [dtheta1, dtheta2]^T

    # each column = how end-effector moves if I rotate that joint by a tiny amount
    t1, t2 = theta
    s1  = np.sin(t1)
    s12 = np.sin(t1 + t2)
    c1  = np.cos(t1)
    c12 = np.cos(t1 + t2)

    J = np.array([
        [-L1 * s1 - L2 * s12,  -L2 * s12],
        [ L1 * c1 + L2 * c12,   L2 * c12]
    ])
    return J

def gravity_torques(theta):
    t1, t2 = theta
    F_grav = np.array([0.0, -g])   # gravity always points down

    # Link 1: center of mass is halfway along the upper arm
    # Jacobian of the link-1 COM with respect to both joints
    # joint 2 doesn't affect link 1's COM, so its column is zero
    J1_com = np.array([
        [-0.5 * L1 * np.sin(t1),  0.0],
        [ 0.5 * L1 * np.cos(t1),  0.0]
    ])
    tau1 = J1_com.T @ (m1 * F_grav)

    # Link 2: center of mass is halfway along the forearm
    # Jacobian of the link-2 COM with respect to both joints
    J2_com = np.array([
        [-L1 * np.sin(t1) - 0.5 * L2 * np.sin(t1 + t2),  -0.5 * L2 * np.sin(t1 + t2)],
        [ L1 * np.cos(t1) + 0.5 * L2 * np.cos(t1 + t2),   0.5 * L2 * np.cos(t1 + t2)]
    ])
    tau2 = J2_com.T @ (m2 * F_grav)

    return tau1 + tau2   # total torque vector [tau_joint1, tau_joint2]

# solve for joint angles using law of cosines, always using CW solution
# fix to not always use CW solution, angle math
def analytic_ik(target):
    x, y = target
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1.0, 1.0)

    t2 = np.arctan2(-np.sqrt(1 - D**2), D)
    t1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))

    return np.array([t1, t2])

def pid_step(theta, theta_dot, theta_target, integral_error):
    e = theta_target - theta    # proportional error
    integral_error = integral_error + e * dt    # accumulate integral
    e_dot = -theta_dot    # derivative of error

    tau = Kp @ e + Ki @ integral_error + Kd @ e_dot

    return tau, integral_error

# Drawing

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
dot_base,  = ax.plot([], [], 'o', color='white',   markersize=10, zorder=5)
dot_elbow, = ax.plot([], [], 'o', color='#aaaaaa', markersize=8,  zorder=5)
dot_end,   = ax.plot([], [], 'o', color='#ffdd00', markersize=9,  zorder=5)

# Target marker
dot_target, = ax.plot([], [], 'x', color='#ff4444',
                      markersize=12, markeredgewidth=2, zorder=6)

# Info text
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

    J = jacobian(theta)
    det = np.linalg.det(J)
    err = np.linalg.norm(target - end) if target is not None else 0.0

    info_text.set_text(
        f"θ₁ = {(np.degrees(theta[0]) % 360):6.1f}°\n"
        f"θ₂ = {(np.degrees(theta[1]) % 360):6.1f}°\n"
        f"det(J) = {det:.3f}\n"
        f"error = {err:.3f}\n"
        f"τ_pid = [{tau_pid[0]:5.2f}, {tau_pid[1]:5.2f}]"
    )

# Animation loop

def animate(_frame):
    global theta, theta_dot, theta_target, target, integral_error

    tau_pid = np.array([0.0, 0.0])

    # 1. Gravity pulls the arm down every frame
    tau_grav = gravity_torques(theta)
    theta_dot += tau_grav * dt

    # 2. PID drives joints toward theta_target (if one is set)
    if theta_target is not None:
        _, _, end = forward_kinematics(theta)
        if np.linalg.norm(target - end) < threshold:
            integral_error = np.array([0.0, 0.0])   # reset integrator on arrival
        tau_pid, integral_error = pid_step(theta, theta_dot, theta_target, integral_error)
        theta_dot += tau_pid * dt

    # 3. Damping, models friction and motor resistance
    theta_dot *= kd

    # 4. Integrate velocity into position
    theta += theta_dot * dt

    update_display(theta, tau_pid)
    return line_upper, line_fore, dot_base, dot_elbow, dot_end, dot_target, info_text

# Mouse click handler

def on_click(event):
    global target, theta_target, integral_error
    if event.inaxes != ax:
        return
    clicked = np.array([event.xdata, event.ydata])
    # Only set target if it's within the reachable workspace
    # range goes beyond L1 + L2 and L1 - L2, look into this
    valid_target = True if (np.linalg.norm(clicked) <= L1 + L2 + 0.05 and np.linalg.norm(clicked) >= L1 - L2 - 0.05) else False
    if valid_target:
        target = clicked
        theta_target = analytic_ik(target)      # solve for joint angles once on click
        integral_error = np.array([0.0, 0.0])   # reset integrator for new target
    else:
        print("Target outside workspace - click between the two circles.")
    print(np.linalg.norm(clicked))

fig.canvas.mpl_connect('button_press_event', on_click)

ani = animation.FuncAnimation(
    fig, animate, interval=30, blit=True, cache_frame_data=False
)

ax.legend(loc='lower right', facecolor='#1a1a1a', labelcolor='white', fontsize=8)
plt.tight_layout()
plt.show()