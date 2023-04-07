import numpy as np
import pickle
import matplotlib.pyplot as plt


filename = r"C:\Users\bardh\OneDrive\programming\control_barrier_function_kit\examples\tutorials\humans_in_scene.pkl"
with open(filename, "rb") as f:
    data = pickle.load(f)


print(data.keys())
t = np.asarray(data["t"])
t -= t[0]
xe = np.array([xx for xx in np.array(data["ego_state"])[:, 1]])
xh = np.array([xx for xx in np.array(data["agent1_state"])[:, 1]])
u = np.array([xx for xx in np.array(data["ego_control"])[:, 1] if len(xx) == 9])

lwidth = 3


ax = plt.subplot(111, xlabel="x (m)", ylabel="y (m)")
for item in (
    [ax.title, ax.xaxis.label, ax.yaxis.label]
    + ax.get_xticklabels()
    + ax.get_yticklabels()
):
    item.set_fontsize(20)


# Corridor Boundaries
xmax = 3.5
xmin = -1.5
ymin = -2.5
ymax = -0.7
ax.plot([xmax, xmax], [ymin, ymax], "k", linewidth=lwidth)
ax.plot([xmin, xmin], [ymin, ymax], "k", linewidth=lwidth)
ax.plot([xmin, xmax], [ymin, ymin], "k", linewidth=lwidth)
ax.plot([xmin, xmax], [ymax, ymax], "k", label="Corridor Boundary", linewidth=lwidth)

# Ego Path
R = 0.25
t1 = 40
t2 = 120
t3 = 150
ax.plot(xe[:, 0], xe[:, 1], "b", label="HSR Agent", linewidth=lwidth)
for tt in range(0, 184, 5):
    ax.add_patch(plt.Circle((xe[tt, 0], xe[tt, 1]), R, color="b", alpha=0.2))
# ax.plot(xe[t1, 0] + R * np.cos(np.linspace(0,2 * np.pi,100)), xe[t1, 1] + R * np.sin(np.linspace(0,2 * np.pi,100)), ':y', linewidth=lwidth)
# ax.plot(xe[t2, 0] + R * np.cos(np.linspace(0,2 * np.pi,100)), xe[t2, 1] + R * np.sin(np.linspace(0,2 * np.pi,100)), ':c', linewidth=lwidth)
# ax.plot(xe[t3, 0] + R * np.cos(np.linspace(0,2 * np.pi,100)), xe[t3, 1] + R * np.sin(np.linspace(0,2 * np.pi,100)), ':m', linewidth=lwidth)

# Agent Path
ax.plot(xh[:, 0], xh[:, 1], "r", label="Human Agent", linewidth=lwidth)
for tt in range(0, 184, 5):
    ax.add_patch(plt.Circle((xh[tt, 0], xh[tt, 1]), R, color="r", alpha=0.2))
# ax.plot(xh[t1, 0] + R * np.cos(np.linspace(0,2 * np.pi,100)), xh[t1, 1] + R * np.sin(np.linspace(0,2 * np.pi,100)), ':y', linewidth=lwidth)
# ax.plot(xh[t2, 0] + R * np.cos(np.linspace(0,2 * np.pi,100)), xh[t2, 1] + R * np.sin(np.linspace(0,2 * np.pi,100)), ':c', linewidth=lwidth)
# ax.plot(xh[t3, 0] + R * np.cos(np.linspace(0,2 * np.pi,100)), xh[t3, 1] + R * np.sin(np.linspace(0,2 * np.pi,100)), ':m', linewidth=lwidth)

# Initial/Goal State
goal_circle = [
    -1.44 + 0.3 * np.cos(np.linspace(0, 2 * np.pi, 100)),
    -2.15 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, 100)),
]
ax.plot(goal_circle[0], goal_circle[1], color="g", label="Goal Set", linewidth=lwidth)
ax.plot(-1.44, -2.15, "*", color="g", markersize=10, label="Goal Point")
ax.plot(xe[0, 0], xe[0, 1], "o", color="m", markersize=10, label="Initial Condition")
ax.plot(xh[0, 0], xh[0, 1], "o", color="m", markersize=10)
ax.plot(xh[-1, 0], xh[-1, 1], "*", color="g", markersize=10)


ax.legend(fontsize=15)


ax21 = plt.subplot(211, xlabel=r"$t$ (s)", ylabel=r"$\omega$ (rad/s)")
for item in (
    [ax21.title, ax21.xaxis.label, ax21.yaxis.label]
    + ax21.get_xticklabels()
    + ax21.get_yticklabels()
):
    item.set_fontsize(20)

ax21.plot(t, u[: len(t), 0])


ax22 = plt.subplot(212, xlabel=r"$t$ (s)", ylabel=r"$a$ (m/s$^2$)")
for item in (
    [ax22.title, ax22.xaxis.label, ax22.yaxis.label]
    + ax22.get_xticklabels()
    + ax22.get_yticklabels()
):
    item.set_fontsize(20)

ax22.plot(t, u[: len(t), 1])

plt.show()

# fig, ax = plt.subplots()

# # Interpolate to ensure same length
# t_new = np.linspace(0, t[-1], len(xe))
# xe_interp = np.vstack([np.interp(t_new, t, xe[:, 0]), np.interp(t_new, t, xe[:, 1])])

# xh_interp = np.vstack(
#     [np.interp(t_new, t, xh[:184, 0]), np.interp(t_new, t, xh[:184, 1])]
# )


# norm_xe_xh = np.linalg.norm(xe_interp - xh_interp, axis=0)
# ax.plot(t_new, norm_xe_xh, label="Norm of HSR and Human Agent")

# # plot horizontal line at 0.25m
# ax.axhline(y=0.5, color="r", linestyle="-", label="Safe Distance Level R=0.5m")
# ax.legend()
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Distance between HSR and Human (m)")

# # reduce white space in plot and print pdf
# fig.tight_layout()
# fig.savefig("dist_hsr_agent.pdf")
# plt.show()
