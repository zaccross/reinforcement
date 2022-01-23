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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


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
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Want to code value iteration for k steps:

        # Put things into common language from class
        k = self.iterations
        gamma = self.discount

        # we want to run the value iteration k times:
        for i in range(k):
            temp_dic = {}
            # Now let's look at all of the possible states:
            states = self.mdp.getStates()

            # Now we need to update all of the values for each state so cycle through:
            for state in states:

                # is this the terminal state
                isEnd = self.mdp.isTerminal(state)
                # get all of the actions so we can find the max
                possible_actions = self.mdp.getPossibleActions(state)
                # This will be our max value that we can update as we cycle through actions
                best_val = None

                # now cycle through and pull out max
                for action in possible_actions:
                    # get new possibilities and set sum = 0
                    total = 0
                    new_chunks = self.mdp.getTransitionStatesAndProbs(state, action)

                    # now sum all of them up
                    for chunk in new_chunks:
                        s_prime = chunk[0]
                        prob = chunk[1]
                        if isEnd:
                            r = 0
                        else:
                            r = self.mdp.getReward(state, action, s_prime)
                        total += prob * (r + gamma * self.values[s_prime])
                    # Keep track of max
                    if best_val is None or total > best_val:
                        best_val = total

                # after all actions:
                if best_val is not None:
                    temp_dic[state] = best_val

                # after testing, the end states need to have a value of 0.
                # makes sense based on them having no possible transitions...
                if isEnd:
                    temp_dic[state] = 0

            # update the dictionary now that the iteration is done
            self.values = temp_dic

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
        "*** YOUR CODE HERE ***"
        # Now to compute q val as sum of all possible states from the actions:
        # the q val is the sum of the states for a give action
        # no real commenting because it follows a similar structure as above
        isEnd = self.mdp.isTerminal(state)

        # catches end terminal state :)
        total = 0

        # Handling the rest similar to above
        possible_chunks = self.mdp.getTransitionStatesAndProbs(state, action)
        for chunk in possible_chunks:
            s_prime = chunk[0]
            prob = chunk[1]
            if isEnd:
                r = 0
            else:
                r = self.mdp.getReward(state, action, s_prime)
            total += prob * (r + self.discount * self.values[s_prime])

        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # check if terminal:
        if self.mdp.isTerminal(state):
            return None
        # again a similar approach as above except now using q-values so no
        # real need to comment

        else:
            possible_actions = self.mdp.getPossibleActions(state)
            best_action = None
            best_val = None
            for action in possible_actions:

                q_val = self.computeQValueFromValues(state, action)
                if best_val is None or q_val > best_val:
                    best_val = q_val
                    best_action = action

            return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """Returns the policy at the state (no exploration)."""
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
        "*** YOUR CODE HERE ***"
        # Put things into common language from class
        k = self.iterations
        gamma = self.discount

        # we want to run the value iteration k times:
        states = self.mdp.getStates()

        # we do K iterations, but only one state at a time
        for i in range(k):

            # modulo opertor so we only get values in states index
            # and loops back after k iterations
            cnt = i % len(states)

            # Now let's look at all of the possible states:
            # Now we need to update all of the values for each state so cycle through:
            state = states[cnt]

            # is this the terminal state
            isEnd = self.mdp.isTerminal(state)

            # get all of the actions so we can find the max
            possible_actions = self.mdp.getPossibleActions(state)

            # This will be our max value that we can update as we cycle through actions
            best_val = None

            # now cycle through and pull out max
            for action in possible_actions:

                # get new possibilities and set sum = 0
                total = 0
                new_chunks = self.mdp.getTransitionStatesAndProbs(state, action)

                # now sum all of them up
                for chunk in new_chunks:
                    s_prime = chunk[0]
                    prob = chunk[1]
                    if isEnd:
                        r = 0
                    else:
                        r = self.mdp.getReward(state, action, s_prime)
                    total += prob * (r + gamma * self.values[s_prime])
                # Keep track of max
                if best_val is None or total > best_val:
                    best_val = total

            # after all actions:
            if best_val is not None:
                self.values[state] = best_val

                # after testing, the end states need to have a value of 0.
                # makes sense based on them having no possible transitions...
            if isEnd:
                self.values[state] = 0


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
        """*** YOUR CODE HERE ***"""
        # following pseudo code in packet

        # part 1 calculate predecessors
        # basically look at each parent's child and put the parent in the
        # child's predecessor entry
        predecessors = {}
        all_states = self.mdp.getStates()

        for state in all_states:

            # use a set to avoid duplicates
            predecessors[state] = set()

        # now do the loop and add predecessors
        # do not have to dp this in real time since the tree is static
        for state in all_states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                possible_states = self.mdp.getTransitionStatesAndProbs(state, action)

                for chunk in possible_states:
                    new_state = chunk[0]
                    predecessors[new_state].add(state)

        # part 2 intialize pqueue
        new_queue = util.PriorityQueue()

        # part 3 pushing state into queue with diff
        # initial issue was not doing abs(diff)
        # want biggest difference to be popped first so put into queue with -diff pri.
        for state in all_states:
            value = self.values[state]

            best_q = None
            new_actions = self.mdp.getPossibleActions(state)

            for action in new_actions:
                new_val = self.computeQValueFromValues(state, action)
                if best_q is None or new_val > best_q:
                    best_q = new_val

            if best_q is not None:
                diff = abs(value - best_q)
                new_queue.update(state, -diff)
            else:
                new_queue.update(state, -abs(value))

        # part 4 iterate
        for i in range(self.iterations):
            if new_queue.isEmpty():
                return None
            else:
                state = new_queue.pop()
                isEnd = self.mdp.isTerminal(state)

                # get all of the actions so we can find the max
                possible_actions = self.mdp.getPossibleActions(state)

                # This will be our max value that we can update as we cycle through actions
                best_val = None

                # now cycle through and pull out max
                for action in possible_actions:
                    # get new possibilities and set sum = 0
                    total = 0
                    new_chunks = self.mdp.getTransitionStatesAndProbs(state, action)

                    # now sum all of them up
                    for chunk in new_chunks:
                        s_prime = chunk[0]
                        prob = chunk[1]
                        if isEnd:
                            r = 0
                        else:
                            r = self.mdp.getReward(state, action, s_prime)
                        total += prob * (r + self.discount * self.values[s_prime])
                    # Keep track of max
                    if best_val is None or total > best_val:
                        best_val = total

                # after all actions:
                if best_val is not None:
                    self.values[state] = best_val

                    # after testing, the end states need to have a value of 0.
                    # makes sense based on them having no possible transitions...

                for p in predecessors[state]:

                    value = self.values[p]

                    best_q = None
                    new_actions = self.mdp.getPossibleActions(p)

                    for action in new_actions:
                        new_val = self.computeQValueFromValues(p, action)
                        if best_q is None or new_val > best_q:
                            best_q = new_val

                    if best_q is not None:
                        diff = abs(value - best_q)
                        if diff > self.theta:
                            new_queue.update(p, -diff)
