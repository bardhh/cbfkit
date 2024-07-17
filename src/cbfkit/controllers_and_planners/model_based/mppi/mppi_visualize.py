import jax.numpy as jnp
import matplotlib.pyplot as plt

def initialize_mppi_plots(ax, plot_num_samples, prediction_horizon, robot_state_dim):
    plot_num_samples = plot_num_samples
    sampled_trajectories_plot = []
    for i in range(plot_num_samples): # sampled trajectories
        sampled_trajectories_plot.append( ax.plot(jnp.ones(prediction_horizon), 0*jnp.ones(prediction_horizon), 'g', alpha=0.2) )
    selected_trajectory_plot = ax.plot(jnp.ones(prediction_horizon), 0*jnp.ones(prediction_horizon), 'b')
    return {
         'robot_state_dim': robot_state_dim,
        'plot_num_samples': plot_num_samples,
        'sampled_trajectories_handle': sampled_trajectories_plot,
        'selected_trajectory_handle': selected_trajectory_plot
    }

def update_mppi_plot(mppi_plots, robot_sampled_states, robot_selected_states):
    for i in range(mppi_plots['plot_num_samples']):
            mppi_plots['sampled_trajectories_handle'][i][0].set_xdata( robot_sampled_states[mppi_plots['robot_state_dim']*i,:] )
            mppi_plots['sampled_trajectories_handle'][i][0].set_ydata( robot_sampled_states[mppi_plots['robot_state_dim']*i+1,:] )
    mppi_plots['selected_trajectory_handle'][0].set_xdata(robot_selected_states[0,:] )
    mppi_plots['selected_trajectory_handle'][0].set_ydata(robot_selected_states[1,:] )