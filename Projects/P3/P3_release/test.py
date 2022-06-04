class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = dict()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState not in predecessors.keys():
                        predecessors[nextState] = set()
                    if prob:
                        predecessors[nextState].add(state)

        priorityQueue = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            diff = abs(self.values[state] - max([self.getQValue(state, action) for action in actions]))
            priorityQueue.push(state, -diff)

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            state = priorityQueue.pop()
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            self.values[state] = max([self.getQValue(state, action) for action in actions])
            for predecessor in predecessors[state]:
                actions = self.mdp.getPossibleActions(predecessor)
                diff = abs(self.values[predecessor] - max([self.getQValue(predecessor, action) for action in actions]))
                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)