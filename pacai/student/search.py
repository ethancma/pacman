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
    frontier.push((start_state, list(), 0), 0)
    explored = set()
    path = list()
    cost_map = dict()

    while not frontier.isEmpty():
        node, action, cost = frontier.pop()
        cost_map[node] = cost

        if problem.isGoal(node):
            path = action
            break

        for s, a, c in problem.successorStates(node):
            if s not in explored:
                frontier.push((s, action + [a], cost + c), cost + c)
                explored.add(s)
            elif s in explored and (s in cost_map and cost_map[s] > (cost + c)):
                frontier.push((s, action + [a], cost + c), cost + c)
                cost_map[s] = cost + c
                explored.add(s)
            explored.add(node)

    return path


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    start_state = problem.startingState()

    frontier = PriorityQueue()
    frontier.push((start_state, list(), 0), 0)
    explored = set()
    path = list()
    cost_map = dict()

    while not frontier.isEmpty():
        node, action, cost = frontier.pop()
        cost_map[node] = cost

        if problem.isGoal(node):
            path = action
            break

        for s, a, c in problem.successorStates(node):
            total_cost = cost + c + heuristic(s, problem)
            if s not in explored:
                frontier.push((s, action + [a], cost + c), total_cost)
                explored.add(s)
            elif s in explored and (s in cost_map and cost_map[s] > total_cost):
                frontier.push((s, action + [a], cost + c), total_cost)
                cost_map[s] = cost + c
                explored.add(s)
            explored.add(node)

    return path
