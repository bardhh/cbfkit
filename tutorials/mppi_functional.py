import jax
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.random import multivariate_normal
from jax import jit, lax

class circle:

    def __init__(self, ax, pos = np.array([0,0]),radius = 1.0):
        self.X = pos.reshape(-1,1)
        self.radius = radius
        self.id = id
        self.type = 'circle'

        self.render(ax)

    def render(self,ax):
        circ = plt.Circle((self.X[0],self.X[1]),self.radius,linewidth = 1, edgecolor='k',facecolor='k')
        ax.add_patch(circ)

test = 1

def setup_mppi_controller(robot_n = 2, robot_m = 2, horizon=10, samples = 10, input_size = 2, control_bound = 2, dt=0.05, u_guess=None, use_GPU=True, costs_lambda = 0.03, num_obstacles = 2, cost_goal_coeff = 0.2, cost_safety_coeff = 10.0, cost_perturbation_coeff=0.1, cost_goal_coeff_final = 0.2, cost_safety_coeff_final = 10.0):

    # self.key = jax.random.PRNGKey(111)
    horizon = horizon
    samples = samples
    robot_m = input_size
    dt = dt
    use_gpu = use_GPU

    robot_n = 2
    robot_m = 2
    control_mu = jnp.zeros(robot_m)#.reshape(-1,1)
    control_cov = 4.0 * jnp.eye(robot_m)
    control_cov_inv = jnp.linalg.inv(control_cov)
    control_bound = control_bound
    control_bound_lb = -jnp.array([1,1]).reshape(-1,1)
    control_bound_ub =  jnp.array([1,1]).reshape(-1,1)
    if u_guess != None:
        U = u_guess
    else:
        U = 3 * jnp.ones((horizon,robot_m))

    costs_lambda = costs_lambda
    cost_goal_coeff = cost_goal_coeff
    cost_safety_coeff = cost_safety_coeff
    cost_perturbation_coeff = cost_perturbation_coeff
    cost_goal_coeff_final = cost_goal_coeff_final
    cost_safety_coeff_final = cost_safety_coeff_final

    num_obstacles = num_obstacles

    @jit
    def robot_dynamics_step(state, input):
        # single integrator
        return state + input * dt
    
        # Unicycle
        theta = state[2,0]
        xdot = jnp.array([ [ input[0,0]*jnp.cos(theta) ],
                            [ input[0,0]*jnp.sin(theta) ],
                            [ input[1,0] ]
                            ])
        return state + xdot * MPPI.dt
    
    @jit
    def weighted_sum(U, perturbation, costs):#weights):
        costs = costs - jnp.min(costs)
        costs = costs / jnp.max(costs)
        lambd = costs_lambda
        weights = jnp.exp( - 1.0/lambd * costs )   # higher cost -> higher weight
        normalization_factor = jnp.sum(weights)
        def body(i, inputs):
            U = inputs
            U = U + perturbation[i] * weights[i] / normalization_factor
            return U
        return lax.fori_loop( 0, samples, body, (U) )

    @jit
    def single_sample_rollout(goal, robot_states_init, perturbed_control, obstaclesX, perturbation):
        
        # Initialize robot_state
        robot_states = jnp.zeros( ( robot_n, horizon) )
        robot_states = robot_states.at[:,0].set(robot_states_init)      

        # loop over horizon
        cost_sample = 0
        # jax.debug.print("🤯 {x} 🤯 {y}", x=goal[0,0], y=goal[1,0])
        def body(i, inputs):
            cost_sample, robot_states, obstaclesX = inputs

            # get robot state
            robot_state = robot_states[:,[i]]

            # Get cost
            cost_sample = cost_sample + cost_goal_coeff * ((robot_state[0:2]-goal).T @ (robot_state[0:2]-goal))[0,0]
            cost_sample = cost_sample + cost_perturbation_coeff  * ((perturbed_control[:, [i]]-perturbation[:,[i]]).T @ control_cov_inv @ perturbation[:,[i]])[0,0]
            robot_obstacle_dists = jnp.linalg.norm(robot_state[0:2] - obstaclesX, axis=0)
            cost_sample = cost_sample + cost_safety_coeff / jnp.max(jnp.array([jnp.min(robot_obstacle_dists)-1.0, 0.01])  )

            # Update robot states
            robot_states = robot_states.at[:,i+1].set( robot_dynamics_step( robot_states[:,[i]], perturbed_control[:, [i]] )[:,0] )
            return cost_sample, robot_states, obstaclesX
        cost_sample, robot_states, _ = lax.fori_loop( 0, horizon-1, body, (cost_sample, robot_states, obstaclesX) )

        robot_state = robot_states[:,[horizon-1]]
        cost_sample = cost_sample + cost_goal_coeff_final * ((robot_state[0:2]-goal).T @ (robot_state[0:2]-goal))[0,0] # goal cost
        cost_sample = cost_sample + cost_perturbation_coeff * ((perturbed_control[:, [horizon]]-perturbation[:,[horizon]]).T @ control_cov_inv @ perturbation[:,[horizon]])[0,0]
        robot_obstacle_dists = jnp.linalg.norm(robot_state[0:2] - obstaclesX, axis=0)
        cost_sample = cost_sample + cost_safety_coeff_final / jnp.max(jnp.array([jnp.min(robot_obstacle_dists)-1.0, 0.01])  )
        
        return cost_sample, robot_states

    @jit
    def rollout_states_foresee(robot_init_state, perturbed_control, goal, obstaclesX, perturbation):

        ##### Initialize
               
        # Robot
        robot_states = jnp.zeros( (samples, robot_n, horizon) )
        robot_states = robot_states.at[:,:,0].set( jnp.tile( robot_init_state.T, (samples,1) ) )

        # Cost
        cost_total = jnp.zeros(samples)

        # Single sample rollout
        if use_gpu:
            @jit
            def body_sample(robot_states_init, perturbed_control_sample, perturbation_sample):
                cost_sample, robot_states_sample = single_sample_rollout(goal, robot_states_init, perturbed_control_sample.T, obstaclesX, perturbation_sample.T )
                return cost_sample, robot_states_sample
            batched_body_sample = jax.vmap( body_sample, in_axes=0 )
            cost_total, robot_states, = batched_body_sample( robot_states[:,:,0], perturbed_control, perturbation)
        else:
            @jit
            def body_samples(i, inputs):
                robot_states, cost_total, obstaclesX = inputs     

                # Get cost
                cost_sample, robot_states_sample = single_sample_rollout(goal, robot_states[i,:,0], perturbed_control[i,:,:].T, obstaclesX, perturbation[i,:,:].T )
                cost_total = cost_total.at[i].set( cost_sample )
                robot_states = robot_states.at[i,:,:].set( robot_states_sample )
                return robot_states, cost_total, obstaclesX  
            robot_states, cost_total, obstaclesX = lax.fori_loop( 0, samples, body_samples, (robot_states, cost_total, obstaclesX) )

        return robot_states, cost_total

    @jit
    def compute_perturbed_control(subkey, control_mu, control_cov, control_bound, U):
        perturbation = multivariate_normal( subkey, control_mu, control_cov, shape=( samples, horizon ) ) # K x T x nu
        
        perturbation = jnp.clip( perturbation, -3.0, 3.0 ) #0.3
        perturbed_control = U + perturbation

        perturbed_control = jnp.clip( perturbed_control, -control_bound, control_bound )
        perturbation = perturbed_control - U
        return perturbation, perturbed_control
    
    @staticmethod
    @jit
    def rollout_control(init_state, actions):
        states = jnp.zeros((robot_n, horizon+1))
        states = states.at[:,0].set(init_state[:,0])
        def body(i, inputs):
            states = inputs
            states = states.at[:,i+1].set( robot_dynamics_step(states[:,[i]], actions[i,:].reshape(-1,1))[:,0] )
            return states
        states = lax.fori_loop(0, horizon, body, states)
        return states
    
    @jit
    def compute_rollout_costs( key, U, init_state, goal, obstaclesX ):

        perturbation, perturbed_control = compute_perturbed_control(key, control_mu, control_cov, control_bound, U)

        sampled_robot_states, costs= rollout_states_foresee(init_state, perturbed_control, goal, obstaclesX, perturbation)

        U = weighted_sum( U, perturbation, costs)

        states_final = rollout_control(init_state, U)              
        action = U[0,:].reshape(-1,1)
        U = jnp.append(U[1:,:], U[[-1],:], axis=0)

        sampled_robot_states = sampled_robot_states.reshape(( robot_n*samples, horizon ))
   
        return sampled_robot_states, states_final, action, U
    
    return compute_rollout_costs
    

if test==1:

    plt.ion()
    fig, ax = plt.subplots( )
    ax.set_xlim([-1,12])
    ax.set_ylim([-1,12])
    dt = 0.05
    tf = 10
    T = int(tf/dt)
    prediction_tf = 2
    prediction_horizon = int(prediction_tf/dt)
    num_samples = 20000
    costs_lambda = 0.03 
    cost_goal_coeff = 0.2
    cost_safety_coeff = 10.0
    cost_goal_coeff_final = 1.0
    cost_safety_coeff_final = 5.0

    # robot
    X = jnp.array([0,0]).reshape(-1,1)
    n = 2
    control_bound = 7

    # obstacle
    radius = 1.0
    obs = np.array([3,3]).reshape(-1,1)
    circ = circle( ax, pos=obs, radius=radius )
    goal = jnp.array([9,9]).reshape(-1,1)
    ax.scatter(goal[0], goal[1], s=50, edgecolors='g', facecolors='none')

    # Setup MPPI controller
    mppi = setup_mppi_controller(horizon=prediction_horizon, samples = num_samples, input_size = 2, control_bound=control_bound, dt=dt, u_guess=None, use_GPU=False, costs_lambda = costs_lambda, cost_goal_coeff = cost_goal_coeff, cost_safety_coeff = cost_safety_coeff, cost_goal_coeff_final = cost_goal_coeff_final, cost_safety_coeff_final = cost_safety_coeff_final, num_obstacles = 1, cost_perturbation_coeff=0.1)
    key = jax.random.PRNGKey(111)

    # Plotting
    robot_body = ax.scatter(X[0,0],X[1,0],c='g',alpha=1.0,s=70)
    plot_num_samples = 10
    sampled_trajectories_plot = []
    for i in range(plot_num_samples): # sampled trajectories
        sampled_trajectories_plot.append( ax.plot(jnp.ones(prediction_horizon), 0*jnp.ones(prediction_horizon), 'g', alpha=0.2) )
    selected_trajectory_plot = ax.plot(jnp.ones(prediction_horizon), 0*jnp.ones(prediction_horizon), 'b')

    # Initial control input guess
    U = 3 * jnp.ones((prediction_horizon,2))

    for t in range(T):
        key, subkey = jax.random.split(key)
        robot_sampled_states, robot_selected_states, robot_action, U = mppi(subkey, U, X, goal, obs)

        X = X + robot_action * dt
        robot_body.set_offsets([X[0,0], X[1,0]])

        for i in range(plot_num_samples):
            sampled_trajectories_plot[i][0].set_xdata( robot_sampled_states[n*i,:] )
            sampled_trajectories_plot[i][0].set_ydata( robot_sampled_states[n*i+1,:] )
        selected_trajectory_plot[0].set_xdata(robot_selected_states[0,:] )
        selected_trajectory_plot[0].set_ydata(robot_selected_states[1,:] )
        print(f"hello")
        fig.canvas.draw()
        fig.canvas.flush_events()



        