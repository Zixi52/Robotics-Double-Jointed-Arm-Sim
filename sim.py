import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Arm params
L1 = 2.0          # length of upper arm
L2 = 1.5          # length of forearm
alpha = 0.1       # IK step size
lam = 0.01        # damping factor for damped least squares
threshold = 0.05  # stop iterating when error is below this

# State
theta = np.array([np.pi / 4, np.pi / 4]) # [theta1, theta2]
target = None # set on mouse click

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
    s1 = np.sin(t1)
    s12 = np.sin(t1 + t2)
    c1 = np.cos(t1)
    c12 = np.cos(t1 + t2)

    J = np.array([
        [-L1 * s1 - L2 * s12,  -L2 * s12],
        [ L1 * c1 + L2 * c12,   L2 * c12]
    ])
    return J

def ik_step(theta, target):
    # One iteration of damped-least-squares Jacobian IK.

    # dtheta = J^T (J J^T + (λ^2)I)^-1 dot e
    _, _, end = forward_kinematics(theta)
    error = target - end

    if np.linalg.norm(error) < threshold:
        return theta, True # reached target

    J = jacobian(theta)
    JJT = J @ J.T
    damped = JJT + lam**2 * np.eye(2)
    delta = J.T @ np.linalg.inv(damped) @ error

    return theta + alpha * delta, False

# Drawing

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-(L1 + L2 + 0.5), L1 + L2 + 0.5)
ax.set_ylim(-(L1 + L2 + 0.5), L1 + L2 + 0.5)
ax.set_aspect('equal')
ax.set_facecolor('#0f0f0f')
fig.patch.set_facecolor('#0f0f0f')
ax.grid(True, color='#333333', linewidth=0.5)
ax.set_title('2D Robotic Arm  —  click to set target', color='white', fontsize=12)
ax.tick_params(colors='#666666')

# Draw workspace boundaries (max and min reach circles)
outer = plt.Circle((0, 0), L1 + L2, color='#333333',
                        fill=False, linestyle='--', linewidth=0.8)
inner = plt.Circle((0, 0), L1 - L2, color='#333333',
                        fill=False, linestyle='--', linewidth=0.8)
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

def update_display(theta):
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
        f"θ₁ = {(np.degrees(theta[0])%360):6.1f}°\n"
        f"θ₂ = {(np.degrees(theta[1])%360):6.1f}°\n"
        f"det(J) = {det:.3f}\n"
        f"error = {err:.3f}"
    )

# Animation loop

def animate(_frame):
    global theta, target
    if target is not None:
        theta, reached = ik_step(theta, target)
        if reached:
            target = None
    update_display(theta)
    return line_upper, line_fore, dot_base, dot_elbow, dot_end, dot_target, info_text

# Mouse click handler

def on_click(event):
    global target
    if event.inaxes != ax:
        return
    clicked = np.array([event.xdata, event.ydata])
    # Only set target if it's within the reachable workspace
    valid_target = True if (np.linalg.norm(clicked) <= L1 + L2 and np.linalg.norm(clicked) >= L1 - L2) else False
    if valid_target:
        target = clicked
    else:
        print("Target outside workspace - click between the two circles.")

fig.canvas.mpl_connect('button_press_event', on_click)

ani = animation.FuncAnimation(
    fig, animate, interval=30, blit=True, cache_frame_data=False
)

ax.legend(loc='lower right', facecolor='#1a1a1a', labelcolor='white', fontsize=8)
plt.tight_layout()
plt.show()