"""Run a discrete-event simulation of an asynchronous evolutionary
algorithm with only cloning and selection, so we can track the
takeover times of particular genomes."""
import inspect
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import toolz

from leap_ec import ops, probe
from leap_ec import Individual, Representation
from leap_ec.algorithm import generational_ea
from leap_ec.problem import ConstantProblem, FunctionProblem
from leap_ec.real_rep import problems as real_prob

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
        max_births = 2
    else:
        jobs = 50
        max_births=float('inf')  # No limit on births; stopping condition is max_time
        
    max_time = 10000
    pop_size = 50
    num_processors=pop_size
    modulo = 10
    low_eval_time = 1
    high_eval_time = 100
    init_strategy = 'immediate'  # Can be 'immediate', 'until_all_evaluating', or 'extra'

    ##############################
    # Problem
    ##############################
    problem = FunctionProblem(lambda x: 0 if x=='LOW' else 100, maximize=True)
    eval_time_prob = FunctionProblem(lambda x: low_eval_time if x=='LOW' else high_eval_time, maximize=True)

    ##############################
    # Setup
    ##############################
    #experiment_note = f"\"takover (low={low_eval_time}, high={high_eval_time})\""
    #experiment_note = init_strategy
    experiment_note = "select(parents+processing)"
    eval_time_f = lambda x: eval_time_prob.evaluate(x)
        
    plt.figure(figsize=(20, 4))

    ##############################
    # Evolve
    ##############################
    with open('birth_times.csv', 'w') as births_file:
        for job_id in range(jobs):

            ##############################
            # Setup Metrics and Simulation
            ##############################
            if gui:
                plt.subplot(144) # Put the Gantt plot on the far right
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
            
            # Set up probes for real-time visualizatiion
            if gui:
                plt.subplot(141)
                p1 = probe.HistPhenotypePlotProbe(ax=plt.gca(), modulo=modulo, title="Genotype Histogram")
                plt.subplot(142)
                p2 = probe.PopulationMetricsPlotProbe(ax=plt.gca(), modulo=modulo,
                                                        title="Fraction of Population with 'HIGH' genotype (by step).",
                                                        metrics=[ lambda x: len([ ind for ind in x if ind.genome == 'HIGH'])/len(x) ])
                plt.subplot(143)
                p3 = probe.PopulationMetricsPlotProbe(ax=plt.gca(), modulo=modulo,
                                                        title="Fraction of Population with 'HIGH' genotype (by time).",
                                                        metrics=[ lambda x: len([ ind for ind in x if ind.genome == 'HIGH'])/len(x) ],
                                                        x_axis_value=lambda: eval_cluster.time)

                # Leave the dashboard in its own window
                p4 = co.AsyncClusterDashboardProbe(cluster_sim=eval_cluster, modulo=modulo)

                gui_probes = [ p1, p2, p3, p4 ]
            else:
                gui_probes = []

            # Defining representation up front, so we can use it a couple different places
            representation=Representation(
                            # Initialize a population of integer-vector genomes
                            initialize=co.single_high_initializer()
                        )

            # GO!
            ea = generational_ea(max_generations=max_births,pop_size=pop_size,

                                    # Stopping condition is based on simulation time
                                    stop=lambda x: eval_cluster.time > max_time,

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
                                                co.select_with_processing(ops.random_selection, eval_cluster),
                                                ops.clone
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
                                                    'LOW_ratio': lambda x: len([ ind for ind in x if ind.genome == 'LOW'])/len(x),
                                                    'HIGH_ratio': lambda x: len([ ind for ind in x if ind.genome == 'HIGH'])/len(x)
                                                }
                                        )
                                    ] + gui_probes
                                )

            # Er, actually go!
            list(ea)
