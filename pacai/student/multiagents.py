import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        score = currentGameState.getScore()

        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()
        newFood = successorGameState.getFood().asList()
        newGhostPos = successorGameState.getGhostPositions()

        foodDist = [distance.manhattan(newPosition, food) for food in oldFood]
        if len(newFood) < len(oldFood):
            score += 1000

        if len(foodDist) >= 2:
            maxFoodDist = max(foodDist)
            minFoodDist = min(foodDist)
            score -= maxFoodDist
            score -= minFoodDist

        ghostDist = [distance.manhattan(newPosition, ghost) for ghost in newGhostPos]
        if min(ghostDist) < 2:
            score -= 1500

        return score


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        legalActions = gameState.getLegalActions()
        value = -9999999
        action = Directions.STOP
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        for a in legalActions:
            successor = gameState.generateSuccessor(0, a)
            temp = self.minValue(successor, 0, 1)
            if temp > value:
                value = temp
                action = a
        return action

    def maxValue(self, s, depth):
        if depth == self.getTreeDepth() or s.isWin() or s.isLose():
            return self.getEvaluationFunction()(s)
        legalActions = s.getLegalActions()
        value = -9999999
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        for action in legalActions:
            successor = s.generateSuccessor(0, action)
            minvalue = self.minValue(successor, depth, 1)
            value = max(value, minvalue)
        return value

    def minValue(self, s, depth, agent):
        if depth == self.getTreeDepth() or s.isWin() or s.isLose():
            return self.getEvaluationFunction()(s)
        legalActions = s.getLegalActions(agent)
        value = 9999999
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        for action in legalActions:
            successor = s.generateSuccessor(agent, action)
            if agent == (s.getNumAgents() - 1):
                value = min(value, self.maxValue(successor, depth + 1))
            else:
                value = min(value, self.minValue(successor, depth, agent + 1))
        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        legalActions = gameState.getLegalActions()
        value = -9999999
        alpha = -9999999
        beta = 9999999
        action = Directions.STOP
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        for a in legalActions:
            successor = gameState.generateSuccessor(0, a)
            temp = self.minValue(successor, 0, 1, alpha, beta)
            if temp > value:
                value = temp
                action = a
        return action

    def maxValue(self, s, depth, alpha, beta):
        if depth == self.getTreeDepth() or s.isWin() or s.isLose():
            return self.getEvaluationFunction()(s)
        legalActions = s.getLegalActions()
        value = -9999999
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        for action in legalActions:
            successor = s.generateSuccessor(0, action)
            minvalue = self.minValue(successor, depth, 1, alpha, beta)
            value = max(value, minvalue)
            # if value >= beta:
            #     return value
            # alpha = max(alpha, value)
            alpha = max(alpha, value)
            if beta <= alpha:
                return value
        return value

    def minValue(self, s, depth, agent, alpha, beta):
        if depth == self.getTreeDepth() or s.isWin() or s.isLose():
            return self.getEvaluationFunction()(s)
        legalActions = s.getLegalActions(agent)
        value = 9999999
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        for action in legalActions:
            successor = s.generateSuccessor(agent, action)
            if agent == (s.getNumAgents() - 1):
                value = min(value, self.maxValue(successor, depth + 1, alpha, beta))
            else:
                value = min(value, self.minValue(successor, depth, agent + 1, alpha, beta))
            # if value <= alpha:
            #     return value
            # beta = min(beta, value)
            beta = min(beta, value)
            if beta <= alpha:
                return value
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        legalActions = gameState.getLegalActions()
        value = -9999999
        action = Directions.STOP
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        for a in legalActions:
            successor = gameState.generateSuccessor(0, a)
            temp = self.expValue(successor, 0, 1)
            if temp > value:
                action = a
                value = temp
        return action

    def maxValue(self, s, depth):
        d = depth + 1
        if d == self.getTreeDepth() or s.isWin() or s.isLose():
            return self.getEvaluationFunction()(s)
        legalActions = s.getLegalActions(0)
        value = -9999999
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        for action in legalActions:
            successor = s.generateSuccessor(0, action)
            v = self.expValue(successor, d, 1)
            value = max(value, v)
        return value

    def expValue(self, s, depth, agent):
        if s.isWin() or s.isLose():
            return self.getEvaluationFunction()(s)
        legalActions = s.getLegalActions(agent)
        if 'Stop' in legalActions:
            legalActions.remove('Stop')
        expValues = 0
        for action in legalActions:
            successor = s.generateSuccessor(agent, action)
            if agent == (s.getNumAgents() - 1):
                value = self.maxValue(successor, depth)
            else:
                value = self.expValue(successor, depth, agent + 1)
            expValues += value
        return 0 if len(legalActions) == 0 else (expValues / len(legalActions))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    score = currentGameState.getScore()

    newPosition = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood().asList()
    newGhostPos = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.getScaredTimer() for ghostState in currentGameState.getGhostStates()]

    foodDist = [distance.manhattan(newPosition, food) for food in oldFood]

    if len(foodDist) >= 2:
        maxFoodDist = max(foodDist)
        minFoodDist = min(foodDist)
        score -= maxFoodDist
        score -= minFoodDist

    ghostDist = [distance.manhattan(newPosition, ghost) for ghost in newGhostPos]
    if min(ghostDist) < 2:
        score -= 1000

    s = sum(scaredTimes)
    if s > 0:
        score += s
    return score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
