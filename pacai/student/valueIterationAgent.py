from pacai.agents.learning.value import ValueEstimationAgent
from copy import deepcopy

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """
    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # Compute the values here.
        # raise NotImplementedError()
        for _ in range(self.iters):
            temp_values = deepcopy(self.values)
            for state in self.mdp.getStates():
                # if self.mdp.isTerminal(state): # works without was getting error earlier
                #     continue
                max_values = [self.getQValue(state, action)
                              for action in self.mdp.getPossibleActions(state)]
                if len(max_values) != 0:
                    temp_values[state] = max(max_values)
            self.values = temp_values

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

    def getPolicy(self, state):
        """Returns the best action in the given state according to computed values."""
        # `pacai.core.mdp.MarkovDecisionProcess.getStates`,
        # `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
        # `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
        # `pacai.core.mdp.MarkovDecisionProcess.getReward`.
        best_action = None
        max_value = -999999
        for action in self.mdp.getPossibleActions(state):
            q_value = self.getQValue(state, action)
            if q_value > max_value:
                max_value = q_value
                best_action = action
        return best_action

    def getQValue(self, state, action):
        """Returns the q-value of the state action pair."""
        q_value = 0
        for next_state, p in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            q_value += p * (reward + self.discountRate * self.getValue(next_state))
        return q_value
