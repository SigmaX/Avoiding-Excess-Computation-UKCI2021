"""Run a discrete-event simulation of an asynchronous evolutionary
algorithm with different fitness landscapes and evaluation-time
functions.
"""
import inspect
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import toolz

from leap_ec import ops, probe
from leap_ec import Representation
from leap_ec.algorithm import generational_ea
from leap_ec.problem import ConstantProblem, FunctionProblem
from leap_ec.real_rep import problems as real_prob
from leap_ec.real_rep import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian

from async_sim import components as co


##############################
# Entry point
##############################
if __name__ == '__main__':
    ##############################
    # Parameters
    ##############################
    gui = True

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(co.test_env_var, False) == 'True':
        jobs = 1
        births = 2
    else:
        jobs = 50
        births=2000

    pop_size = 10
    num_processors=pop_size
    dim = 10
    modulo = 10
    init_strategy = 'immediate'  # Can be 'immediate', 'until_all_evaluating', or 'extra'

    problem_name = 'exponential-growth'
    eval_time_name = 'exponential-growth'
    problem = co.get_standard_function(problem_name, dimensions=dim)
    eval_time_prob = co.get_standard_function(eval_time_name, dimensions=dim)

    ##############################
    # Setup
    ##############################
    #experiment_note = f"\"{problem_name} fitness\n{eval_time_name} eval-time (no crossover)\""
    experiment_note = init_strategy
    eval_time_f = lambda x: eval_time_prob.evaluate(x)
    
    plt.figure(figsize=(20, 8))

    ##############################
    # For each job
    ##############################
    with open('birth_times.csv', 'w') as births_file:
        for job_id in range(jobs):

            ##############################
            # Setup Metrics and Simulation
            ##############################
            if gui:  # Set up the top row of visuals
                pad_val = 0  # Value to fix higher-dimensional values at when projecting landscapes into 2-D visuals
                plt.subplot(231, projection='3d')
                real_prob.plot_2d_problem(problem, xlim=problem.bounds, ylim=problem.bounds, pad=np.array([pad_val]*(dim - 2)), title="Fitness Landscape", ax=plt.gca())

                plt.subplot(232, projection='3d')
                real_prob.plot_2d_function(eval_time_f, xlim=problem.bounds, ylim=problem.bounds, pad=np.array([pad_val]*(dim - 2)), title="Eval-Time Landscape", ax=plt.gca())

                plt.subplot(233)  # Put the Gantt plot in the upper-right
                p = co.GanttPlotProbe(ax=plt.gca(), max_bars=100, modulo=modulo)

                gui_steadystate_probes = [ p ]
            else:
                gui_steadystate_probes = []
            
            # Set up the cluster simulation.
            # This mimics an asynchronous evaluation engine, which may return individuals in an order different than they were submitted.
            eval_cluster = co.AsyncClusterSimulation(
                                            num_processors=num_processors,
                                            eval_time_function=eval_time_f,
                                            # Individual-level probes (these run just on the newly evaluated individual)
                                            probes=[
                                                probe.AttributesCSVProbe(attributes=['birth', 'start_time', 'end_time', 'eval_time'],
                                                                         notes={ 'job': job_id }, header=(job_id==0), stream=births_file)
                                            ] + gui_steadystate_probes)

            # Set up the second row of visuals
            if gui:
                plt.subplot(234)
                p1 = probe.CartesianPhenotypePlotProbe(xlim=problem.bounds, ylim=problem.bounds,
                                                contours=problem, pad=np.array([pad_val]*(dim - 2)), ax=plt.gca(), modulo=modulo)
                
                plt.subplot(235)
                p2 = probe.FitnessPlotProbe(ax=plt.gca(), modulo=modulo,
                                            title="Best-of-step Fitness (by step).")
                
                plt.subplot(236)
                p3 = probe.FitnessPlotProbe(ax=plt.gca(), modulo=modulo,
                                            title="Best-of-step Fitness (by time).",
                                            x_axis_value=lambda: eval_cluster.time)

                # Leave the dashboard in its own window
                p4 = co.AsyncClusterDashboardProbe(cluster_sim=eval_cluster, modulo=modulo)

                gui_pop_probes = [ p1, p2, p3, p4 ]
            else:
                gui_pop_probes = []

            # Defining representation up front, so we can use it a couple different places
            representation=Representation(
                            # Initialize a population of integer-vector genomes
                            initialize=create_real_vector(bounds=[problem.bounds] * dim)
                        )

            ##############################
            # Evolve
            ##############################
            ea = generational_ea(max_generations=births,pop_size=pop_size,
                                    
                                    # We use an asynchronous scheme to evaluate the initial population
                                    init_evaluate=co.async_init_evaluate(
                                        cluster_sim=eval_cluster,
                                        strategy=init_strategy,
                                        create_individual=lambda: representation.create_individual(problem)),
                                        
                                    problem=problem,  # Fitness function

                                    # Representation
                                    representation=representation,

                                    # Operator pipeline
                                    pipeline=[
                                        co.steady_state_step(
                                            reproduction_pipeline=[
                                                #ops.random_selection,
                                                co.select_with_processing(ops.random_selection, eval_cluster),  # SWEET
                                                ops.clone,
                                                #ops.uniform_crossover(p_swap=0.2),
                                                mutate_gaussian(std=1.5, hard_bounds=[problem.bounds]*dim,
                                                        expected_num_mutations=1)
                                            ],
                                            insert=co.competition_inserter(p_accept_even_if_worse=0.0, pop_size=pop_size,
                                                replacement_selector=ops.random_selection
                                            ),
                                            # This tells the steady-state algorithm to use asynchronous evaluation
                                            evaluation_op=co.async_evaluate(cluster_sim=eval_cluster)
                                        ),
                                        # Population-level probes (these run on all individuals in the population)
                                        probe.FitnessStatsCSVProbe(stream=sys.stdout, header=(job_id==0), modulo=modulo,
                                                comment=inspect.getsource(sys.modules[__name__]),  # Put the entire source code in the comments
                                                notes={'experiment': experiment_note, 'job': job_id},
                                                extra_metrics={
                                                    'time': lambda x: eval_cluster.time, 
                                                    'birth': lambda x: eval_cluster.birth,
                                                    'mean_eval_time': lambda x: np.mean([ind.eval_time for ind in x]),
                                                    'diversity': probe.pairwise_squared_distance_metric
                                                }
                                        ),
                                    ] + gui_pop_probes
                                )

            # Er, actually go!
            list(ea)
