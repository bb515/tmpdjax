import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as random
from scipy.stats import wasserstein_distance
import numpy as np
import scipy
import matplotlib.animation as animation


BG_ALPHA = 1.0
MG_ALPHA = 1.0
FG_ALPHA = 0.3
color_posterior = '#a2c4c9'
color_algorithm = '#ff7878'
dpi_val = 1200
cmap = 'magma'


def plot_animation(fig, ax, animate, frames, fname, fps=20, bitrate=800, dpi=300):
  ani = animation.FuncAnimation(
    fig, animate, frames=frames, interval=1, fargs=(ax,))
  # Set up formatting for the movie files
  Writer = animation.PillowWriter
#   Writer = animation.FFMpegWriter
  # To save the animation using Pillow as a gif
  writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=bitrate)
  # Note that mp4 does not work on pdf
#   ani.save('{}.mp4'.format(fname), writer=writer)
  ani.save('{}.gif'.format(fname), writer=writer, dpi=dpi)


def plot_single_image(noise_std, dim, dim_y, timesteps, i, name, indices, samples, color=color_algorithm):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(*samples[:, indices].T, alpha=.5, color=color, edgecolors="black", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-22, 22])
    ax.set_ylim([-22, 22])
    fig.subplots_adjust(left=.005, right=.995,
                        bottom=.005, top=.995)
    plt.savefig(
        'inverse_problem_comparison_out_dist_{}_{}_{}_{}_{}_{}.pdf'.format(noise_std, dim, dim_y, timesteps, i, name), dpi=dpi_val)
    plt.savefig(
        'inverse_problem_comparison_out_dist_{}_{}_{}_{}_{}_{}.png'.format(noise_std, dim, dim_y, timesteps, i, name), transparent=True, dpi=dpi_val)
    plt.close(fig)


def plot_image(noise_std, dim, dim_y, timesteps, i, name, indices, diffusion_samples, target_samples=None):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(*target_samples[:, indices].T, alpha=.5, color=color_posterior, edgecolors= "black", rasterized=True)
    ax.scatter(*diffusion_samples[:, indices].T, alpha=.5, color=color_algorithm, edgecolors="black", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-22, 22])
    ax.set_ylim([-22, 22])
    fig.subplots_adjust(left=.005, right=.995,
                        bottom=.005, top=.995)
    plt.savefig(
        'inverse_problem_comparison_out_dist_{}_{}_{}_{}_{}_{}.pdf'.format(noise_std, dim, dim_y, timesteps, i, name), dpi=dpi_val)
    plt.savefig(
        'inverse_problem_comparison_out_dist_{}_{}_{}_{}_{}_{}.png'.format(noise_std, dim, dim_y, timesteps, i, name), transparent=True, dpi=dpi_val)
    plt.close(fig)


def sliced_wasserstein(rng, dist_1, dist_2, n_slices=100):
    projections = random.normal(rng, (n_slices, dist_1.shape[1]))
    projections = projections / jnp.linalg.norm(projections, axis=-1)[:, None]
    dist_1_projected = (projections @ dist_1.T)
    dist_2_projected = (projections @ dist_2.T)
    return np.mean([wasserstein_distance(u_values=d1, v_values=d2) for d1, d2 in zip(dist_1_projected, dist_2_projected)])


def Wasserstein2(m1, C1, m2, C2):
    C1_half = scipy.linalg.sqrtm(C1)
    C_half = jnp.asarray(np.asarray(np.real(scipy.linalg.sqrtm(C1_half @ C2 @ C1_half)), dtype=float))
    return jnp.linalg.norm(m1 - m2)**2 + jnp.trace(C1) + jnp.trace(C2) - 2 * jnp.trace(C_half)


def Distance2(m1, C1, m2, C2):
    C2_half = jnp.linalg.cholesky(C2)
    C1_half = jnp.linalg.cholesky(C1)
    return jnp.linalg.norm(m1 - m2)**2 + jnp.linalg.norm(C1_half - C2_half)**2


def plot(train_data, test_data, mean, variance,
         fname="plot.png"):
    X, y = train_data
    X_show, f_show, variance_show = test_data
    # Plot result
    fig, ax = plt.subplots(1, 1)
    ax.plot(X_show, f_show, label="True", color="orange")
    ax.plot(X_show, mean, label="Prediction", linestyle="--", color="blue")
    ax.scatter(X, y, label="Observations", color="black", s=20)
    ax.fill_between(
        X_show.flatten(), mean - 2. * jnp.sqrt(variance),
        mean + 2. * jnp.sqrt(variance), alpha=FG_ALPHA, color="blue")
    ax.fill_between(
        X_show.flatten(), f_show - 2. * jnp.sqrt(variance_show),
        f_show + 2. * jnp.sqrt(variance_show), alpha=FG_ALPHA, color="orange")
    ax.set_xlim((X_show[0], X_show[-1]))
    ax.set_ylim((-2.4, 2.4))
    ax.grid(visible=True, which='major', linestyle='-')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(BG_ALPHA)
    ax.patch.set_alpha(MG_ALPHA)
    ax.legend()
    fig.savefig(fname)
    plt.close()


def plot_beta_schedule(sde, solver):
    """Plots the temperature schedule of the SDE marginals.

    Args:
        sde: a valid SDE class.
    """
    beta_t = sde.beta_min + solver.ts * (sde.beta_max - sde.beta_min)
    diffusion = jnp.sqrt(beta_t)

    plt.plot(solver.ts, beta_t, label="beta_t")
    plt.plot(solver.ts, diffusion, label="diffusion_t")
    plt.legend()
    plt.savefig("plot_beta_schedule.png")
    plt.close()
