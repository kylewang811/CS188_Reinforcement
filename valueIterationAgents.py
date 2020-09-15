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
        for i in range(self.iterations):
            copy_of_values = self.values.copy()

            for j in self.mdp.getStates():
                answer = -1000000

                for k in self.mdp.getPossibleActions(j):
                    temp = self.computeQValueFromValues(j, k)
                    answer = max(answer, temp)

                if answer == -1000000:
                    copy_of_values[j] = 0
                else:
                    copy_of_values[j] = answer

            self.values = copy_of_values

    def getValue(self, state):
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        answer = 0
        for i, j in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, i)
            discount = self.discount * self.getValue(i)
            answer += j * (reward + discount)
        return answer

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        moves = self.mdp.getPossibleActions(state)
        if moves:
            reward = -1000000
            answer = None
            for i in moves:
                temp = self.getQValue(state, i)
                if temp > reward:
                    reward = temp
                    answer = i
            return answer
        return None

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
        "*** YOUR CODE HERE ***"
        possible_states = self.mdp.getStates()
        for i in range(self.iterations):
            mod = i % len(possible_states)
            if self.mdp.isTerminal(possible_states[mod]):
                continue
            else:
                answer = -1000000
                moves = self.mdp.getPossibleActions(possible_states[mod])
                for i in moves:
                    temp = self.computeQValueFromValues(possible_states[mod], i)
                    answer = max(temp, answer)
                if answer == -1000000:
                    self.values[possible_states[mod]] = 0
                else:
                    self.values[possible_states[mod]] = answer

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
        "*** YOUR CODE HERE ***"
        def helper_function(x):
            answer = -1000000
            for i in self.mdp.getPossibleActions(x):
                answer = max(answer, self.getQValue(x, i))
            if answer == -1000000:
                return 0
            return answer

        array = {}
        #Initialization of empty sets
        for i in self.mdp.getStates():
            array[i] = set()

        for i in self.mdp.getStates():
            if not self.mdp.isTerminal(i):
                for j in self.mdp.getPossibleActions(i):
                    for x, y in self.mdp.getTransitionStatesAndProbs(i, j):
                        if y > 0:
                            array[x].add(i)
            else:
                continue

        queue = util.PriorityQueue()
        for i in self.mdp.getStates():
            if not self.mdp.isTerminal(i):
                temp = self.getValue(i) - helper_function(i)
                if temp < 0:
                    queue.push(i, temp)
                else:
                    queue.push(i, -temp)
            else:
                continue

        for i in range(self.iterations):
            if not queue.isEmpty():
                temp = queue.pop()
                if not self.mdp.isTerminal(temp):
                    self.values[temp] = helper_function(temp)

                for j in array[temp]:
                    difference = self.values[j] - helper_function(j)
                    if difference > self.theta or difference < -self.theta:
                        if difference < 0:
                            queue.update(j, difference)
                        else:
                            queue.update(j, -difference)
            else:
                i += self.iterations
