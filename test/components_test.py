from leap_ec import Individual
from leap_ec.decoder import IdentityDecoder
from leap_ec.problem import ConstantProblem
import numpy as np
import toolz

from async_sim import components as co


##############################
# Tests for AsyncClusterSimulation
##############################
def test_asynchronousclustersimulation():
    """If we submit a number of individuals and then wait for them
    all to evaluate, they should come back in the order we'd expect from
    their start & evaluation times.
    """
    async_sim = co.AsyncClusterSimulation(num_processors=2, eval_time_function=lambda x: x)

    inds = [ Individual(1, problem=ConstantProblem()),
             Individual(3, problem=ConstantProblem()),
             Individual(4, problem=ConstantProblem()),
             Individual(4, problem=ConstantProblem())
    ]

    for ind in inds:
        async_sim.submit(ind)
    assert(async_sim.num_waiting == 2)
    assert(async_sim.num_processing == 2)

    # The submission time should be recorded as 0 for all four jobs
    for ind in inds:
        assert(ind.submitted_time == 0)

    # The first to come back should be the shortest running one
    result = async_sim.get_next_evaluated()
    assert(result.genome == 1)
    assert(result.end_time == 1)
    assert(async_sim.time == 1)
    assert(async_sim.num_waiting == 1)
    assert(async_sim.num_processing == 2)

    # At this point, the third job should have started
    assert(inds[2].start_time == 1)

    # Next to finish be the longer of the first two jobs
    result = async_sim.get_next_evaluated()
    assert(result.genome == 3)
    assert(result.end_time == 3)
    assert(async_sim.time == 3)
    assert(async_sim.num_waiting == 0)
    assert(async_sim.num_processing == 2)

    # Pulling that will have allowed the 4th job to start
    assert(inds[3].start_time == 3)

    # Now, the third individual takes 4s, but it started at 1s, so it should come out at t=5s:
    result = async_sim.get_next_evaluated()
    assert(result.genome == 4)
    assert(result.start_time == 1)
    assert(result.end_time == 5)
    assert(async_sim.time == 5)
    assert(async_sim.num_waiting == 0)
    assert(async_sim.num_processing == 1)

    # Add another job.  It will start immediately since there is a free processor.
    fifth_ind = Individual(1, problem=ConstantProblem())
    async_sim.submit(fifth_ind)
    assert(fifth_ind.submitted_time == 5)
    assert(fifth_ind.start_time == 5)
    assert(async_sim.time == 5)
    assert(async_sim.num_waiting == 0)
    assert(async_sim.num_processing == 2)
    
    # The fourth individual to come back will be the one we just added, because it's a lot faster than ind[3].
    result = async_sim.get_next_evaluated()
    assert(result is fifth_ind)
    assert(result.start_time == 5)
    assert(result.end_time == 6)
    assert(async_sim.time == 6)
    assert(async_sim.num_waiting == 0)
    assert(async_sim.num_processing == 1)

    # Finally, ind[3] comes out.  It started at 3s and took 4s, landing us at t=7s.
    result = async_sim.get_next_evaluated()
    assert(result is inds[3])
    assert(result.start_time == 3)
    assert(result.end_time == 7)
    assert(async_sim.time == 7)
    assert(async_sim.num_waiting == 0)
    assert(async_sim.num_processing == 0)


##############################
# Tests for async_init_evaluate()
##############################
def test_async_init_evaluate1():
    """
    If we've got 20 individuals and 10 processors and all individuals take 1s to evaluate, and the population
    size is twice the number of processors, then 'until_all_evaluating' intialization should take 2 simulated seconds.
    """
    pop_size = 20
    num_processors = 10

    problem = ConstantProblem()
    population = [ Individual(genome=i, decoder=IdentityDecoder(), problem=problem) for i in range(pop_size) ]

    cluster_sim = co.AsyncClusterSimulation(
        num_processors=num_processors,
        eval_time_function=lambda x: 1,
    )

    evaluated_pop = co.async_init_evaluate(population, cluster_sim=cluster_sim, strategy='until_all_evaluating')
    assert(len(evaluated_pop) == pop_size - (num_processors - 1))
    assert(cluster_sim.num_processing == num_processors - 1)
    assert(cluster_sim.time == 2)


def test_async_init_evaluate2():
    """
    After initializing a population with N = 20 individuals over T = 10 processors
    with the 'until_all_evaluating' strategy, we should have N - (T - 1) = 11 individuals
    in the population, T - 1 = 9 still processing, and nothing in the wait_queue (since
    'until_all_evaluating' doesn't use the wait_queue).
    """
    pop_size = 20
    num_processors = 10

    problem = ConstantProblem()

    # Initialize populations with random phenomes
    population = [ Individual(genome=np.random.uniform(0, 100), problem=problem) for _ in range(pop_size) ]

    cluster_sim = co.AsyncClusterSimulation(
        num_processors=num_processors,
        eval_time_function=lambda x: x,  # Evaluation time is equal to the phenome
    )

    # Evaluate the population
    evaluated_pop = co.async_init_evaluate(population, cluster_sim=cluster_sim, strategy='until_all_evaluating')

    # Check that our state matches what we expected from the 'until_all_evaluating' strategy
    assert(len(evaluated_pop) == pop_size - (num_processors - 1))
    assert(cluster_sim.num_processing == num_processors - 1)
    assert(cluster_sim.num_waiting == 0)


def test_async_init_evaluate3():
    """
    When initializing a population with the 'until_all_evaluating' strategy, and the number of
    processors equals the population size, then the time elapsed at the end of initialization
    should be equal to the shortest individual's eval time.

    (This is because we evaluate just enough individuals to free up 1 processor, which in this
    case means waiting for just 1 individual to come back from evaluating.)
    """
    pop_size = 20
    num_processors = pop_size

    problem = ConstantProblem()

    # Initialize populations with random phenomes
    population = [ Individual(genome=np.random.uniform(0, 100), problem=problem) for _ in range(pop_size) ]

    cluster_sim = co.AsyncClusterSimulation(
        num_processors=num_processors,
        eval_time_function=lambda x: x,  # Evaluation time is equal to the phenome
    )

    # Evaluate the population
    evaluated_pop = co.async_init_evaluate(population, cluster_sim=cluster_sim, strategy='until_all_evaluating')

    fastest = min(evaluated_pop)
    assert(cluster_sim.time == fastest.genome)


def test_async_init_evaluate4():
    """When using the 'immediate' initializations strategy, there should be exactly one individual
    in the population when initialization is complete, and N - 1 individuals processing or in 
    the wait_queue."""
    pop_size = 20
    num_processors = 10

    problem = ConstantProblem()

    # Initialize populations with random phenomes
    population = [ Individual(genome=np.random.uniform(0, 100), problem=problem) for _ in range(pop_size) ]

    cluster_sim = co.AsyncClusterSimulation(
        num_processors=num_processors,
        eval_time_function=lambda x: x,  # Evaluation time is equal to the phenome
    )

    # Evaluate the population
    evaluated_pop = co.async_init_evaluate(population, cluster_sim=cluster_sim, strategy='immediate')

    # Check that our state matches what we expected from the 'immediate' strategy
    assert(len(evaluated_pop) == 1)
    assert(cluster_sim.num_processing == num_processors)
    assert(cluster_sim.num_processing + cluster_sim.num_waiting == pop_size - 1)

def test_async_init_evaluate4():
    """When using the 'extra' initializations strategy, there should be N individuals
    in the population when initialization is complete, nothing in the wait_queue,
    and T - 1 individuals processing (i.e. one open processor)."""
    pop_size = 20
    num_processors = 10

    problem = ConstantProblem()

    # Initialize populations with random phenomes
    create = lambda: Individual(genome=np.random.uniform(0, 100), problem=problem)
    population = [ create() for _ in range(pop_size) ]

    cluster_sim = co.AsyncClusterSimulation(
        num_processors=num_processors,
        eval_time_function=lambda x: x,  # Evaluation time is equal to the phenome
    )

    # Evaluate the population
    evaluated_pop = co.async_init_evaluate(population,
                                           cluster_sim=cluster_sim,
                                           strategy='extra',
                                           create_individual=create)

    # Check that our state matches what we expected from the 'immediate' strategy
    assert(len(evaluated_pop) == pop_size)
    assert(cluster_sim.num_waiting == 0)
    assert(cluster_sim.num_processing == num_processors - 1)
