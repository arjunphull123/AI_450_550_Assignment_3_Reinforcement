# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for _ in range(self.iterations):  # iterate
            values = util.Counter()  # create an empty counter
            for state in self.mdp.getStates():  # iterate across states
                if not self.mdp.isTerminal(state):  # if the state isn't terminal
                    # get the max q value
                    values[state] = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
            # update values
            self.values = values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qVal = 0  # initialize a q value

        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):  # get next state and proability
            reward = self.mdp.getReward(state, action, nextState)  # get reward
            qVal += prob * (reward + self.discount * self.values[nextState])  # compute q value

        return qVal

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):  # if there are no legal actions
            return None
        
        bestAction = None  # initialize a best action
        maxQ = float('-inf')  # and a max value

        for action in self.mdp.getPossibleActions(state):  # iterate across all actions
            qVal = self.computeQValueFromValues(state, action)  # get the q value
            if qVal > maxQ:  # if this is a higher q value
                bestAction = action  # get the action (note that this chooses the first occurrence in case of tie)
                maxQ = qVal

        return bestAction
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()  # get states
        for i in range(self.iterations):  # iterate
            state = states[i % len(states)]  # get the current state based on iteration
            if not self.mdp.isTerminal(state):  # if the state isn't terminal
                # get the max 
                self.values[state] = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])

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
        # compute predecessors of all states
        states = self.mdp.getStates()
        predecessors = {state: set() for state in states}  # dictionary mapping states to sets of predecessors

        for state in states:  # loop through states
            for action in self.mdp.getPossibleActions(state):  # loop through actions
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):  # loop through successors
                    if prob > 0:  # if possible successor:
                        predecessors[nextState].add(state)  # then the state is a predecessor

        # initialize empty priority queue
        pQueue = util.PriorityQueue()

        # iterate over non-terminal states
        for s in states:
            if not self.mdp.isTerminal(s):
                maxQ = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])  # get the max Q val
                diff = abs(self.values[s] - maxQ)  # calculate the difference
                pQueue.update(s, -diff)  # update the state with -diff (de facto max queue)

        # for iterations:
        for _ in range(self.iterations):

            # terminate if the queue is empty
            if pQueue.isEmpty():
                break

            # pop a state
            s = pQueue.pop()
            if not self.mdp.isTerminal(s):
                # update the value if it is not terminal
                self.values[s] = max([self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)])
            
            # iterate across predecessors
            for p in predecessors[s]:
                maxQ = max([self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)])  # get max q
                diff = abs(self.values[p] - maxQ)  # calculate difference

                if diff > self.theta:
                    pQueue.update(p, -diff)  # update predecessor