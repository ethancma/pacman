"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    start_state = problem.startingState()

    frontier = Stack()
    frontier.push((start_state, list()))
    explored = set()
    path = list()

    while not frontier.isEmpty():
        node, c_action = frontier.pop()

        if problem.isGoal(node):
            path = c_action
            break

        for state, action, _ in problem.successorStates(node):
            if state not in explored:
                frontier.push((state, c_action + [action]))
                explored.add(state)
        explored.add(node)

    return path


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    start_state = problem.startingState()

    frontier = Queue()
    frontier.push((start_state, list()))
    explored = set()
    path = list()

    while not frontier.isEmpty():
        node, c_action = frontier.pop()

        if problem.isGoal(node):
            path = c_action
            break

        for state, action, _ in problem.successorStates(node):
            if state not in explored:
                frontier.push((state, c_action + [action]))
                explored.add(state)
        explored.add(node)

    return path

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    start_state = problem.startingState()

    frontier = PriorityQueue()
    explored = list()
    path = list()

    return path


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()
