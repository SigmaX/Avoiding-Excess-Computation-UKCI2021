"""
This module contains reusable components that play a role in our experiments.

Basically, these are our custom extensions to LEAP to allow certain non-standard
algorithms and measurements.
"""
from queue import PriorityQueue, SimpleQueue

from leap_ec import context
from leap_ec import ops
from leap_ec.problem import ConstantProblem, FunctionProblem, ScalarProblem
from leap_ec.real_rep import problems as real_prob
from matplotlib import pyplot as plt
import numpy as np
import toolz


##############################
# test_env_var
##############################
# Name of the environment variable we look for to see if we are running system tests
test_env_var = 'ASYNC_SIM_TESTING'


##############################
# Class AsyncClusterSimulation()
##############################
class AsyncClusterSimulation():
    """
    A queue-based discrete-event simulation of asynchronous fitness
    evaluation on a cluster of nodes.

    This works by jumping forward to the next event in the simulation—
    namely, each completed fitness evaluation—in simulated time.

    This evaluates individuals all on one thread—but it *pretends* that
    each evaluation takes a certain amount of *time* according to some
    distribution (given by `eval_time_function`), and that it has 
    `num_processors` processors to distribute evaluation across.

    This allows us to to produce asynchronous evolutionary algorithms 
    whose search trajectory behaves as it would if we assume the
    given eval-time distribution, but which run very fast (given a
    fitness function that is, in reality, very cheap).
    """
    def __init__(self, num_processors: int, eval_time_function, probes=(), allow_wait_queue: bool = True):
        assert(num_processors > 0)
        assert(eval_time_function is not None)
        self.num_processors = num_processors
        self.eval_time_function = eval_time_function
        self.probes = probes
        self.allow_wait_queue = allow_wait_queue
        self.reset()
        assert(self._rep_okay())

    @property
    def processors_are_full(self) -> bool:
        return self.processing_queue.full()

    @property
    def num_processing(self) -> int:
        return self.processing_queue.qsize()

    @property
    def num_waiting(self) -> int:
        return self.wait_queue.qsize()

    @property
    def processing(self) -> list:
        """Return a list of all the individuals that are currently processing.
        
        For example, if we start a cluster and wait for 1 job to complete after submitting
        3 individuals...

        >>> from leap_ec import Individual
        >>> async_sim = AsyncClusterSimulation(num_processors=2, eval_time_function=lambda x: 1)
        >>> async_sim.submit(Individual([1, 2, 3], problem=real_prob.SpheroidProblem()))
        >>> async_sim.submit(Individual([4, 5, 6], problem=real_prob.SpheroidProblem()))
        >>> finished = async_sim.get_next_evaluated()
        >>> async_sim.submit(Individual([7, 8, 9], problem=real_prob.SpheroidProblem()))

        then the second and third individual we submitted will be the ones that are currently
        processing:

        >>> async_sim.processing #doctest: +ELLIPSIS
        [Individual([4, 5, 6], ...), Individual([7, 8, 9], ...)]

        
        """
        l = []
        for t, ind in self.processing_queue.queue:
            l.append(ind)
        return l

    def reset(self):
        """Reset the state of this class to its initial condition (clearing all queues and counters).
        
        For example, if we submit a number of jobs like so:

        >>> import numpy as np
        >>> from leap_ec import Individual
        >>> async_sim = AsyncClusterSimulation(num_processors=2, eval_time_function=lambda x: np.sum(np.power(x, 2)))
        >>> async_sim.submit(Individual([1, 2, 3], problem=real_prob.SpheroidProblem()))
        >>> async_sim.submit(Individual([4, 5, 6], problem=real_prob.SpheroidProblem()))
        >>> async_sim.submit(Individual([7, 8, 9], problem=real_prob.SpheroidProblem()))
        >>> async_sim.submit(Individual([3, 2, 1], problem=real_prob.SpheroidProblem()))

        Then we built up quite a bit of state:

        >>> async_sim.num_processing, async_sim.num_waiting, async_sim.birth
        (2, 2, 4)

        Reseting it make it like it never happened:

        >>> async_sim.reset()
        >>> async_sim.num_processing, async_sim.num_waiting, async_sim.birth
        (0, 0, 0)
        """
        self.processing_queue = PriorityQueue(maxsize=self.num_processors)
        self.wait_queue = SimpleQueue()
        self.birth = 0
        self.time = 0
        assert(self._rep_okay())

    def submit(self, ind):
        """Submit an individual to the cluster for evaluation. If a node is available, 
        evaluation will begin immediately; otherwise it will be placed in the `wait_queue`.
        
        For example, if we add two individuals to this simulation

        >>> import numpy as np
        >>> from leap_ec import Individual
        >>> async_sim = AsyncClusterSimulation(num_processors=2, eval_time_function=lambda x: np.sum(np.power(x, 2)))
        >>> async_sim.submit(Individual([1, 2, 3], problem=real_prob.SpheroidProblem()))
        >>> async_sim.submit(Individual([4, 5, 6], problem=real_prob.SpheroidProblem()))

        then they will both be evaluated immediately:

        >>> async_sim.num_processing, async_sim.num_waiting
        (2, 0)

        But if we add one more, then it will go to the wait_queue:

        >>> async_sim.submit(Individual([7, 8, 9], problem=real_prob.SpheroidProblem()))

        >>> async_sim.num_processing, async_sim.num_waiting
        (2, 1)
        """
        ind.submitted_time = self.time
        ind.birth = self.birth
        self.birth += 1

        if self.processors_are_full:
            if self.allow_wait_queue:
                self.wait_queue.put(ind)
            else:
                raise ValueError(f"Attempted to submit {ind}, but all processors are full and the wait_queue is disabled.")
        else:
            self._start_evaluation(ind)
            
        assert(self._rep_okay())

    def _start_evaluation(self, ind):
        """Insert an individual into the processing_queue and record its start and end time."""
        assert(self.processing_queue.qsize() < self.processing_queue.maxsize), f"Attempted to insert {ind} into processing_queue, but the queue is full ({self.processing_queue.qsize()} individuals are already being evaluated)."
        ind.start_time = self.time  # Stamp individual with its start time, for recording purposes
        ind.eval_time = self.eval_time_function(ind.decode())
        ind.end_time = self.time + ind.eval_time
        self.processing_queue.put( (ind.end_time, ind) )

    def get_next_evaluated(self):
        """Retrieve the next individual from the priority queue,
        assign it a fitness value, and record its evalution time.
        
        For example, here we set up a simulation where evaluation time is equal to
        the square of the genotype:

        >>> import numpy as np
        >>> async_sim = AsyncClusterSimulation(num_processors=2, eval_time_function=lambda x: np.sum(np.power(x, 2)))

        Now if we submit two individuals to be evaluated that have different evaluation times:

        >>> from leap_ec import Individual
        >>> async_sim.submit(Individual([200], problem=real_prob.SpheroidProblem()))
        >>> async_sim.submit(Individual([10], problem=real_prob.SpheroidProblem()))

        then the one with the shorter evaluation time will be returned first:

        >>> ind = async_sim.get_next_evaluated()
        >>> ind.genome == [ 10 ]
        True

        And it'll have been stamped with several bits of book-keeping.
        >>> ind.eval_time, ind.submitted_time, ind.start_time, ind.end_time
        (100, 0, 0, 100)
        """
        assert(self.processing_queue.qsize() > 0), "Attempted to wait for the next evaluation to complete, but the processors are empty."
        end_time, ind = self.processing_queue.get()
        assert(ind.end_time == ind.start_time + ind.eval_time)
        assert(ind.end_time >= self.time), f"Received an individual at t={self.time} that ended in the past (start time {ind.start_time}, end time {ind.end_time})..."

        # Advance the clock
        self.time = ind.end_time

        ind.evaluate()  # Do actual fitness evaluation here

        # Run probes to collect data
        for p in self.probes:
            p([ind])

        # Start evaluating the next item in the queue
        if not self.wait_queue.empty():
            self._start_evaluation(self.wait_queue.get())

        assert(self._rep_okay())
        return ind

    def _rep_okay(self):
        """Representation invariant: should return true before and after all public method calls."""
         # Nothing should be waiting in the queue if we disallow waiting, and
         # nothing should be waiting in the queue if we have free processors
        return (self.wait_queue.empty() or self.allow_wait_queue) and \
            (self.wait_queue.empty() or self.processors_are_full) 


##############################
# Class AsyncClusterDashboardProb
##############################
class AsyncClusterDashboardProbe():
    """
    A bar plot of the jobs currently processing in the cluster.
    """
    def __init__(self, cluster_sim, ax=None, title='Async Cluster Status',
                 modulo=1, context=context):
        assert(cluster_sim is not None)
        self.cluster_sim = cluster_sim

        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        
        ax.set_title(title)
        self.title = title
        self.modulo = modulo
        self.context = context

    def __call__(self, population):
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']
        
        # Update the plot
        if step % self.modulo == 0:
            self.ax.cla()
            
            for i, (_, ind) in enumerate(self.cluster_sim.processing_queue.queue):
                s, e = ind.start_time, ind.eval_time
                t = self.cluster_sim.time - s
                self.ax.broken_barh([(s, e)], (i, 0.9), facecolors='gray')
                self.ax.broken_barh([(s, t)], (i, 0.9), facecolors='blue')

            self.ax.set_title(self.title)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Processor")
            #self.ax.figure.canvas.draw()
            plt.pause(0.000001)

        return population


##############################
# Function async_init_evaluate
##############################
@toolz.curry
def async_init_evaluate(population, cluster_sim, strategy: str, create_individual=None):
    """Evaluates an initial population, by sending individuals
    out to processors for evaluation, and building a new
    population by adding individuals in the order that they
    complete evaluating.

    This operator is meant to be used at initialization time, to
    asynchronously initialize a population in preparation for
    steady-state evolution.
    
    The population returned here is not full: after this
    function completes, there will still be `num_processors - 1`
    individuals evaluating on the processors.
    
    The result is a simulation with exactly one free processor—
    ready to be filled with an offspring individual when
    steady-state evolution begins."""
    assert(cluster_sim.num_processors <= len(population))

    next_individual = iter(population)

    if strategy == 'immediate':
        # Strategy 1: wait until the *first* individual comes back, and then
        # end (so we can begin breeding immediately from the partially-filled population)
        for ind in next_individual:
            cluster_sim.submit(ind)
        new_pop = [ cluster_sim.get_next_evaluated() ]

    elif strategy == 'until_all_evaluating':
        # Strategy 2: wait until all individuals are either evaluated or actively being
        # evaluated, then end (so we can begin breeding on the first free processor).

        # First fill up all but one processor
        for _ in range(cluster_sim.num_processors - 1):
            ind = next(next_individual)
            cluster_sim.submit(ind)
        assert(cluster_sim.num_waiting == 0)
        assert(cluster_sim.num_processing == cluster_sim.num_processors - 1)
        
        # Send off the rest of the population using steady-state logic
        evaluated = async_evaluate(next_individual, cluster_sim=cluster_sim)
        new_pop = list(evaluated)
        assert(len(new_pop) == len(population) - (cluster_sim.num_processors - 1)), f"After initialization, we have {cluster_sim.num_processing} individuals processing and {len(new_pop)} completed; but expected {len(population) - (cluster_sim.num_processors - 1)} completed."
   
    elif strategy == 'extra':
        # Strategy 3: generate a few extra individuals to keep the processors busy
        # while we begin steady-state evolution

        if create_individual is None:
            raise ValueError(f"The {strategy} initialization strategy was selected, but no create_individual() method was given.")

        # Submit the initial sample
        for ind in next_individual: 
            cluster_sim.submit(ind)

        # And a few extra to keep things busy
        for _ in range(cluster_sim.num_processors - 1):
            new_ind = create_individual()
            cluster_sim.submit(new_ind)
        assert(cluster_sim.num_processing == cluster_sim.num_processors)
        assert(cluster_sim.num_waiting == (len(population) - 1))
        
        # Fill the population up to the desired size
        new_pop = [ cluster_sim.get_next_evaluated() for _ in range(len(population))]

        # Now nothing is waiting and we have one processor free, ready for steady-state evolution
        assert(cluster_sim.num_waiting == 0)
        assert(cluster_sim.num_processing == (cluster_sim.num_processors - 1))

    else:
        raise ValueError(f"Unrecognized initialization strategy {strategy}.")

    return new_pop


##############################
# Function async_evaluate()
##############################
@toolz.curry
def async_evaluate(next_individual, cluster_sim):
    """Sends individuals off to be evaluated asynchronously,
    and returns the next individuals to be completed.

    This operator is meant to be used with `steady_state_step`
    to plug into the regular evolutionary portion of an EA.
    
    The output of this function depends on what is currently 
    in the processing queue: we return what *finishes* evaluating,
    but this may be a different individual than the one we just
    *sent* to be evaluated.
    """
    for ind in next_individual:
        cluster_sim.submit(ind)
        yield cluster_sim.get_next_evaluated()


##############################
# Function steady_state_step()
##############################
@toolz.curry
@ops.listlist_op
def steady_state_step(population: list, reproduction_pipeline: list, insert, probes = (), evaluation_op = ops.evaluate):
    """An operator that performs steady-state evolution when placed in an (otherwise
    generational) pipeline.

    This is a metaheuristic component that can be parameterized to define many kinds of 
    steady-state evolution.  It takes a population, uses the `reproduction_pipeline` to
    produce a single new individual, evaluates it with the provided `evaluation_op` operator,
    and then inserts the individual returned by `evaluation_op` into the population using the
    strategy defined by `insert`.
    """
    offspring = next(toolz.pipe(population, *reproduction_pipeline))
    evaluated = next(evaluation_op(iter([ offspring ])))
    new_pop = insert(population, evaluated)

    # Run the probes on the ind regardless of the result of insert()
    list(toolz.pipe(iter([evaluated]), *probes))
    return new_pop


##############################
# Function competition_inserter
##############################
@toolz.curry
def competition_inserter(population, new_individual, p_accept_even_if_worse: float, replacement_selector, pop_size: int = None):
    assert(replacement_selector is not None)

    new_pop = population[:]

    if (pop_size is not None) and (len(population) < pop_size):
        # The population isn't full—so just insert without competing
        new_pop.append(new_individual)
    else:
        # Choose a competitor
        indices = []
        competitor = next(replacement_selector(population, indices=indices))
        assert(len(indices) == 1)
        competitor_index = indices[0]

        # Accept new_ind if it's better or the dice rolls that way
        accept = np.random.uniform(0, 1) < p_accept_even_if_worse
        if accept or new_individual > competitor:
            new_pop[competitor_index] = new_individual
    
    return new_pop


##############################
# Function select_with_processing()
##############################
def select_with_processing(selector, cluster_sim):
    """A selection wrapper that uses the given operator to select from 
    the union of the population and the currently processing individuals.
    """
    def select(population):
        return selector(population + cluster_sim.processing)
    return select


##############################
# Class TwoBasinProblem
##############################
class TwoBasinProblem(ScalarProblem):
    """A simple fitness function with two local optima each surrounded by a 
    Gaussian basin of attraction.
    """
    def __init__(self, a: float, b: float, dimensions: int, maximize=False):
        assert(dimensions > 0)
        self.a, self.b = a, b
        self.dimensions = dimensions
        self.maximize = maximize
        self.basin_1 = real_prob.TranslatedProblem(
            problem=real_prob.GaussianProblem(width=1, height=1, maximize=maximize),
            offset=[-1]*dimensions,
            maximize=False
        )
        self.basin_2 = real_prob.TranslatedProblem(
            problem=real_prob.GaussianProblem(width=1, height=1, maximize=maximize),
            offset=[1]*dimensions,
            maximize=False
        )

    @property
    def bounds(self):
        return [-2, 2]

    def evaluate(self, phenome):
        assert(len(phenome) == self.dimensions), f"Got {len(phenome)} dimensions, expected {self.dimensions}."
        y_offset = -min(self.a, self.b) if self.a < 0 or self.b < 0 else 0
        return y_offset + self.a*self.basin_1.evaluate(phenome) + self.b*self.basin_2.evaluate(phenome)


##############################
# Function get_standard_function()
##############################
def get_standard_function(name: str, dimensions: int):
    """Select from a few preset functions we use in these experiments."""
    if name == 'exponential-growth':
        problem = FunctionProblem(lambda x: np.exp(np.sum(x)), maximize=True)
        problem.bounds = [-2, 2]
        return problem
    if name == 'exponential-decay':
        problem = FunctionProblem(lambda x: np.exp(-np.sum(x)), maximize=True)
        problem.bounds = [-2, 2]
        return problem
    if name == 'constant':
        problem = ConstantProblem()
        problem.bounds = [-2, 2]
        return problem
    if name == 'two-basin-equal':
        return TwoBasinProblem(a=1, b=1, dimensions=dimensions, maximize=True)
    if name == 'two-basin-unequal':
        return TwoBasinProblem(a=1, b=10, dimensions=dimensions, maximize=True)
    if name == 'random-uniform':
        problem = FunctionProblem(lambda x: np.random.uniform(), maximize=True)
        problem.bounds = [-2, 2]
        return problem


##############################
# Class GanttPlotProbe
##############################
class GanttPlotProbe():
    """
    A Gantt-chart plot that displays the start and end time of individual
    fitness evaluations.
    """
    def __init__(self, start_f=lambda x: x.start_time, time_f=lambda x: x.eval_time,
                 birth_f=lambda x: x.birth, ax=None, title='Gantt Chart',
                 max_bars = None, modulo=1, context=context):
        assert(callable(start_f))
        assert(callable(time_f))
        self.start_f = start_f
        self.time_f = time_f
        self.birth_f = birth_f

        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        
        ax.set_title(title)
        self.title = title
        self.max_bars = max_bars
        self.modulo = modulo
        self.context = context

        # Some local state
        self.intervals = {}

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']
            
        # Collect data to plot
        for ind in population:
            s, t = self.start_f(ind), self.time_f(ind)
            b = self.birth_f(ind)
            if b not in self.intervals.keys():
                self.intervals[b] = (s, t)

        # Update the plot
        if step % self.modulo == 0:
            self.ax.cla()

            max_b = max(self.intervals.keys())
            min_b = max(0, max_b - self.max_bars) if self.max_bars else 0
            
            for b, (s, t) in list(self.intervals.items()):
                if b > min_b:
                    self.ax.broken_barh([(s, t)], (b, 0.9))
                else:
                    del self.intervals[b]

            self.ax.set_title(self.title)
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Generations")
            #self.ax.figure.canvas.draw()
            plt.pause(0.000001)

        return population


##############################
# Initializer
##############################
def single_high_initializer(low_genome='LOW', high_genome='HIGH'):
    """Generate a population that contains two types of genomes:
    all of the population will be 'LOW' except for one, which will
    be 'HIGH'."""
    first = True

    def create():
        nonlocal first
        if first:
            first = False
            return high_genome
        else:
            return low_genome
    
    return create