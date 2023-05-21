from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection, probability
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.reinforcementagent.getlegalactions`:
    get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipcoin`:
    flip a coin (get a binary value) with some probability.

    `random.choice`:
    pick randomly from a list.

    additional methods to implement:

    `pacai.agents.base.baseagent.getaction`:
    compute the action to take in the current state.
    with probability `pacai.agents.learning.reinforcement.reinforcementagent.getepsilon`,
    we should take a random action and take the best policy action otherwise.
    note that if there are no legal actions, which is the case at the terminal state,
    you should choose none as the action.

    `pacai.agents.learning.reinforcement.reinforcementagent.update`:
    the parent class calls this to observe a state transition and reward.
    you should do your q-value update here.
    note that you should never call this function, it will be called on your behalf.

    description: <write something here so we know what you did.>
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        # you can initialize q-values here.
        self.q_values = dict()

    def getQValue(self, state, action):
        """
        get the q-value for a `pacai.core.gamestate.abstractgamestate`
        and `pacai.core.directions.directions`.
        should return 0.0 if the (state, action) pair has never been seen.
        """
        return self.q_values[(state, action)] if (state, action) in self.q_values else 0.0

    def getValue(self, state):
        """
        return the value of the best action in a state.
        i.e., the value of the action that solves: `max_action q(state, action)`.
        where the max is over legal actions.
        note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        this method pairs with `qlearningagent.getpolicy`,
        which returns the actual best action.
        whereas this method returns the value of the best action.
        """
        return max([self.getQValue(state, action) for action in self.getLegalActions(state)],
                   default=0.0)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        max_qvalue = -999999
        best_actions = None
        if len(self.getLegalActions(state)) == 0:
            return None
        for action in self.getLegalActions(state):
            q_val = self.getQValue(state, action)
            if q_val > max_qvalue:
                max_qvalue = q_val
                best_actions = action
            elif max_qvalue == q_val:
                best_actions = random.choice([best_actions, action])
        return best_actions  # random tie break for better behavior

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a state transition and reward.
        You should do your Q-Value update here.
        Note that you should never call this function, it will be called on your behalf.
        """
        q = self.getQValue(state, action)
        tra = self.getAlpha() * (reward + self.getDiscountRate() * self.getValue(nextState) - q)
        k = (state, action)
        self.q_values[k] = self.getQValue(state, action) + tra

    def getAction(self, state):
        """
        Compute the action to take in the current state.
        With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
        we should take a random action and take the best policy action otherwise.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should choose None as the action.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        prob = probability.flipCoin(1 - self.getEpsilon())
        action = self.getPolicy(state) if prob else random.choice(legalActions)
        return action

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            raise NotImplementedError()
