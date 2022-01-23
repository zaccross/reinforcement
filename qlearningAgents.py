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
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math


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
        "*** YOUR CODE HERE ***"
        # sets the discount, epsilon ... etc
        ReinforcementAgent.__init__(self, **args)

        # initialized !
        self.q_values = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # if the Q state is in the dict, return the val, otherwise add it in and initialize to 0
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        else:
            self.q_values[(state, action)] = 0
            return 0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Get legal actions
        actions = self.getLegalActions(state)

        # return 0 if there are no legal actions
        if len(actions) == 0:
            return 0

        # store best q value to find max
        best_q_value = None

        # loop through legal actions and compare q values
        for action in actions:

            # use our previous function to get the q val
            q_val = self.getQValue(state, action)

            # make sure the key is in there
            if best_q_value is None or q_val > best_q_value:
                best_q_value = q_val

        return best_q_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # get all actions and return None if there aren't any
        actions = self.getLegalActions(state)

        if len(actions) == 0:
            return None

        # store best q value to find max
        best_q_value = self.getValue(state)
        best_actions = []

        # loop through legal actions and compare q values
        for action in actions:

            # make a q state
            q_val = self.getQValue(state, action)

            # if we have the best them add the action into the list
            # this way we can break ties using a random choice from the best_actions list
            if q_val == best_q_value:
                best_actions.append(action)

        return random.choice(best_actions)

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
        "*** YOUR CODE HERE ***"
        # Get all of the legal actions
        legal_actions = self.getLegalActions(state)
        best_action = self.computeActionFromQValues(state)

        # This is where the noise comes in
        # flip a coin with prob epsilon that we draw a random action
        # if True then we pick a legal action at random and if False then we go with the best.
        random_prob = util.flipCoin(self.epsilon)
        if random_prob:
            return random.choice(legal_actions)

        else:
            return best_action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Follows formula from packet
        # calculate the q val of next state
        best_q_prime = self.getValue(nextState)

        # calculate the sample
        sample = reward + (self.discount * best_q_prime)

        # get the current value
        currentQ = self.getQValue(state, action)

        # write down the q_state to use as a key and do the update
        q = (state, action)
        self.q_values[q] = ((1 - self.alpha) * currentQ) + (self.alpha * sample)

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
        "*** YOUR CODE HERE ***"

        # Each qvalue is the dot product of the feautures and weights
        # set inital val to 0 and then go through and do the dot product
        q = 0
        feature_vector = self.featExtractor.getFeatures(state, action)
        for component in feature_vector:
            q += self.weights[component] * feature_vector[component]
        return q

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # use codeish in the packet
        # Similar to getQval above.
        # get the q val and the next one and update the weights according to packet equation

        feauture_vector = self.featExtractor.getFeatures(state, action)
        q_val = self.getQValue(state, action)
        q_prime = self.computeValueFromQValues(nextState)
        difference = (reward + self.discount*q_prime) - q_val

        for component in feauture_vector:
            current_weight = self.weights[component]
            self.weights[component] = current_weight + self.alpha*difference*feauture_vector[component]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"

            print(self.weights)
            pass
