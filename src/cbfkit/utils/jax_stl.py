import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

import time


# Processes mtl not
@jit
def jax_not(robustness):
    return -robustness


# Processes mtl or
@jit
def jax_or(left_robustness, right_robustness):
    return jnp.maximum(left_robustness, right_robustness)


# Processes mtl and
@jit
def jax_and(left_robustness, right_robustness):
    return jnp.minimum(left_robustness, right_robustness)


# Binary search for a desired time step
@jit
def search_sorted(time_stamps, time):  # , start_lower_index):

    def search_body(i, bounds):
        lower, upper, middle = bounds
        middle = (lower + upper) // 2
        condition = time_stamps[middle] <= time
        lower = lax.select(condition, middle + 1, lower)
        upper = lax.select(
            ~condition & (time_stamps[middle] != time),
            middle - 1,
            upper,
        )
        return lower, upper, middle

    lower, upper, middle = 0, time_stamps.shape[0] - 1, 0
    lower, upper, middle = lax.fori_loop(
        0,
        jnp.ceil(jnp.log2(time_stamps.shape[0])).astype(int),
        search_body,
        (lower, upper, middle),
    )
    return middle


# Returns the max of two inputs
@jit
def max_jax(left, right):
    return jnp.maximum(left, right)


# Returns the min of two inputs
@jit
def min_jax(left, right):
    return jnp.minimum(left, right)


# Finds the index of the minimum value in array
@jit
def find_min(array, start_index, end_index):
    def min_body(i, min_idx):
        new_min_idx = lax.cond(
            array[start_index + i] < array[min_idx],
            lambda _: start_index + i,
            lambda _: min_idx,
            None,
        )
        return new_min_idx

    return lax.fori_loop(0, end_index - start_index + 1, min_body, start_index)


# Finds the index of the max value in array
@jit
def find_max(array, start_index, end_index):
    def max_body(i, max_idx):
        new_max_idx = lax.cond(
            array[start_index + i] > array[max_idx],
            lambda _: start_index + i,
            lambda _: max_idx,
            None,
        )
        return new_max_idx

    return lax.fori_loop(0, end_index - start_index + 1, max_body, start_index)


# Custom accumulate function using lax.scan
@jit
def accumulate_max(arr):
    def body(carry, x):
        return jnp.maximum(carry, x), jnp.maximum(carry, x)

    _, result = lax.scan(body, arr[0], arr)
    return result


@jit
def accumulate_min(arr):
    def body(carry, x):
        return jnp.minimum(carry, x), jnp.minimum(carry, x)

    _, result = lax.scan(body, arr[0], arr)
    return result


# finally can also be within a time interval only


# Processes mtl finally
@jit
def jax_finally(lower_time_bound, upper_time_bound, robustness, time_stamps):
    length = len(robustness)

    def process_valid_bounds(_):
        def inf_case(_):
            # In case the upper time bound is infinity, we accumulate maximum robustness
            return jnp.max(robustness)

        def bounded_case(_):
            upper_bound_index = search_sorted(time_stamps, upper_time_bound)
            # jax.debug.print("🤯 upper index {x} 🤯", x=upper_bound_index)
            lower_bound_index = lax.cond(
                lower_time_bound == 0,
                lambda _: 0,
                lambda _: search_sorted(time_stamps, lower_time_bound),
                None,
            )
            # jax.debug.print("🤯 lower index {x} 🤯", x=lower_bound_index)
            # Check if indices are equal and return corresponding robustness
            return lax.cond(
                lower_bound_index == upper_bound_index,
                lambda _: robustness[lower_bound_index],
                lambda _: robustness[find_max(robustness, lower_bound_index, upper_bound_index)],
                None,
            )
            # def compute_finally(i):

        return lax.cond(
            (lower_time_bound == 0) & jnp.isinf(upper_time_bound),
            inf_case,
            bounded_case,
            None,
        )

    robustness = lax.cond(
        upper_time_bound > time_stamps[0], process_valid_bounds, lambda _: 0.0, None
    )  # valid

    return robustness


# Processes mtl global
# #@jit
def jax_global(lower_time_bound, upper_time_bound, robustness, time_stamps):
    length = len(robustness)

    def process_valid_bounds(_):

        def inf_case(_):
            # In case the upper time bound is infinity, we accumulate minimum robustness
            return jnp.min(robustness)

        def bounded_case(_):
            upper_bound_index = search_sorted(time_stamps, upper_time_bound)
            lower_bound_index = lax.cond(
                lower_time_bound == 0,
                lambda _: 0,
                lambda _: search_sorted(time_stamps, lower_time_bound),
                None,
            )

            # Check if indices are equal and return corresponding robustness
            return lax.cond(
                lower_bound_index == upper_bound_index,
                lambda _: robustness[lower_bound_index],
                lambda _: robustness[find_min(robustness, lower_bound_index, upper_bound_index)],
                None,
            )

        return lax.cond(
            (lower_time_bound == 0) & jnp.isinf(upper_time_bound),
            inf_case,
            bounded_case,
            None,
        )

    robustness = lax.cond(
        upper_time_bound > time_stamps[0], process_valid_bounds, lambda _: 0.0, None  # valid
    )

    return robustness


# Processes 1-dimensional polyhedron A*trace[i] <= bound
@jit
def jax_one_dim_pred(traces, A, bound):
    return -1 * (traces * A - bound)


# Processes mtl until
@jit
def jax_until(upper_time_bound, robustness, time_stamps):

    robustness = lax.cond(
        upper_time_bound > time_stamps[0],
        jax_global,
        lambda x, y, z, w: 0.0,
        time_stamps[0],
        upper_time_bound,
        robustness,
        time_stamps,  # valid
    )

    return robustness


# Processes mtl next
@jit
def jax_next(robustness):
    return jnp.roll(robustness, -1)


# Example usage
if __name__ == "__main__":
    length = 5  # 0000
    key = jax.random.PRNGKey(0)
    robustness = jax.random.uniform(key, shape=(length,))
    robustness = jnp.array([0.57450044, 0.8941783, 0.09968603, 0.39316022, 0.59656656])
    print(f"robustness: {robustness}")
    time_stamps = jnp.linspace(1, 10, length)
    print(f"timestamps: {time_stamps}")

    execution_times = []

    # JIT compile the function
    jax_global_jit = jit(jax_global)

    # Run the function once to ensure it's compiled
    val = jax_global_jit(0, jnp.inf, robustness, time_stamps)
    _ = jax_global_jit(0, jnp.inf, robustness, time_stamps)
    # print(f"global : {jax_global(0, jnp.inf, robustness, time_stamps)}")
    # print(f"finally : {jax_finally(0, jnp.inf, robustness, time_stamps)}")
    # print(f"until : {jax_until(jnp.inf, robustness, time_stamps)}")

    # print(f"index : {jnp.argmax(time_stamps>=3.5)}")
    # print(f"global : {jax_global(2.0, jnp.inf, robustness, time_stamps)}")
    # print(f"finally : {jax_finally(5.1, 7.4, robustness, time_stamps)}")
    print(f"until : {jax_until(0.6, robustness, time_stamps)}")

    # start_time = time.time()

    # # Using JAX's lax.fori_loop for optimized looping
    # def timing_loop(_, times):
    #     start = time.time()
    #     val = jax_global_jit(3, 5, robustness, time_stamps)
    #     end = time.time()
    #     execution_time = end - start
    #     return times.at[_].set(execution_time)

    # execution_times = lax.fori_loop(0, 1000000, timing_loop, jnp.zeros(100))

    # total_time = time.time() - start_time

    # min_time = jnp.min(execution_times)
    # max_time = jnp.max(execution_times)
    # avg_time = jnp.mean(execution_times)

    # print(f"Total execution time for 100 iterations: {total_time:.6f} seconds")
    # print(f"Minimum execution time for a single iteration: {min_time:.6f} seconds")
    # print(f"Maximum execution time for a single iteration: {max_time:.6f} seconds")
    # print(f"Average execution time per iteration: {avg_time:.6f} seconds")
    # print(f"Standard deviation of execution times: {jnp.std(execution_times):.6f} seconds")
    # print(f"Number of iterations: {len(execution_times)}")
    # print(f"Length of the trajectory: {length}")
