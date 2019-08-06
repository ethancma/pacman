import random

import pacai.util.util

from pacai.agents.base import BaseAgent
from pacai.core.game import Directions

class LeftTurnAgent(BaseAgent):
    """
    An agent that turns left at every opportunity
    """

    def __init__(self, index):
        super().__init__(index)

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH

        left = Directions.LEFT[current]
        if left in legal:
            return left

        if current in legal:
            return current

        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]

        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]

        return Directions.STOP

class GreedyAgent(BaseAgent):
    def __init__(self, index, evalFn = "scoreEvaluation"):
        super().__init__(index)

        self.evaluationFunction = pacai.util.util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]

        return random.choice(bestActions)

def scoreEvaluation(state):
    return state.getScore()