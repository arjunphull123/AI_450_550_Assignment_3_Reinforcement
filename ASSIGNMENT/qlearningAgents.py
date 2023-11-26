# qlearningAgents.py
# ------------------
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


from game import *
from featureExtractors import *
from learningAgents import ReinforcementAgent
import util
import random


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qvalues = dict()  # (state, action): q-value

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (state, action) in self.qvalues:  # if seen before
            return self.qvalues[(state, action)]  # get Q node value
        return 0.0  # otherwise return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)  # get legal actions
        if not legalActions:  # if no legal actions
            return 0.0
        return max([self.getQValue(state, action) for action in legalActions])  # get maximum q value across legal actions

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)

        if not legalActions:  # terminal state
            return None
        
        maxQ = float('-inf')  # initialize value for max Q
        bestActions = []  # initialize a list of best actions (for the case of a tie)

        for action in legalActions:  # iterate over legal actions
            qVal = self.getQValue(state, action)  # get the q value
            if qVal > maxQ:
                maxQ = qVal
                bestActions = [action]  # update the best actions list
            elif qVal == maxQ:  # in the case of a tie
                bestActions.append(action)
        
        return random.choice(bestActions)  # for ties, chooses a random action. otherwise returns the best action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if legalActions:  # if there are actions to choose from
            if util.flipCoin(self.epsilon):  # take a random action
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)  # take the best policy action

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # get maxQ
        maxQ = 0  # initialize max Q for next state
        if self.getLegalActions(nextState):  # if not terminal
            maxQ = max([self.getQValue(nextState, a) for a in self.getLegalActions(nextState)])  # get the max 1
        q = self.getQValue(state, action)  # get current q
        q_new = (1 - self.alpha) * q + self.alpha * (reward + self.discount * maxQ)  # update the q value
        self.qvalues[(state, action)] = q_new  # store the q value in dictionary

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        qVal = 0  # initialize a q value
        featureVector = self.featExtractor.getFeatures(state, action)  # get the feature vector
        for feature, value in featureVector.items():  # iterate across the feature vector and its values
            w = self.weights[feature]  # get the corresponding weight
            qVal += w * value  # increment q value by the product
        return qVal  # the dot product of weights and featureVector

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # get max q value for next state
        maxQNext = 0  # initialize
        if self.getLegalActions(nextState):
            maxQNext = max([self.getQValue(nextState, action) for action in self.getLegalActions(nextState)])

        diff = (reward + self.discount * maxQNext) - self.getQValue(state, action)  # calculate diff
        featureVector = self.featExtractor.getFeatures(state, action)  # get feature vector

        for feature, value in featureVector.items():  # iterate across feature vector
            w = self.weights[feature]
            self.weights[feature] = w + self.alpha * diff * value  # update weights one by one

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
