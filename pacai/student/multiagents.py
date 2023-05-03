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

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()
        newFood = successorGameState.getFood().asList()
        # newGhostStates = successorGameState.getGhostStates()
        newGhostPos = successorGameState.getGhostPositions()

        # *** Your Code Here ***
        # get manhattan distance from position to foods
        # get manhattan distance from position to ghosts

        # closest ghost
        # closest food
        score = 0
        foodDist = [distance.manhattan(newPosition, foodPos) for foodPos in oldFood]
        ghostDist = [distance.manhattan(newPosition, ghostPos) for ghostPos in newGhostPos]
        if action == 'Stop':
            score -= 500

        closestFood = min(foodDist, default=0)
        closestGhost = min(ghostDist, default=0)
        if closestGhost == 0:
            closestGhost = 0.001
        if closestGhost < 2:
            score -= 1000
        if len(newFood) < len(oldFood):
            score += 1000
        if closestFood == 0:
            score += closestGhost * score

        score += (closestFood * 1000) / (closestGhost) * score

        return successorGameState.getScore() + score

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

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()

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
