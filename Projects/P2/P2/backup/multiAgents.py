from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # --- info of next state, after moving --- #
        newPos = successorGameState.getPacmanPosition() # pacman position
        newFood = successorGameState.getFood() # remaining food
        newFoodList = newFood.asList() # food position list
        newCapture = successorGameState.getCapsules() # captures postion list
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] 
        newGhostPos = successorGameState.getGhostPositions() # ghost position

        # --- info of current state, before moving --- #
        curPos = currentGameState.getPacmanPosition() # pacman position
        curFood = currentGameState.getFood() # remaining food
        curFoodList = curFood.asList() # food position list
        curCapsules = currentGameState.getCapsules() # captures postion list
        curGhostStates = currentGameState.getGhostStates()
        curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates] 
        curGhostPos = currentGameState.getGhostPositions() # ghost position

        "*** YOUR CODE HERE ***"
        score = 0 # scoring indicator
        eatAdot = 10 # reward unit
        powerFlags = [] # whether if you eat a capture

        for curScaredTime in curScaredTimes:
            if (curScaredTime != 0):
                powerFlags.append(True)
            else:
                powerFlags.append(False)

        # distance to ghost
        i = 0
        for powerflag in powerFlags:
            # once eat a capture, try to eat ghost~
            if (powerflag):
                if (manhattanDistance(newPos, curGhostPos[i]) < curScaredTimes[i]):
                    if (manhattanDistance(newPos, curGhostPos[i]) == 0):
                        score += 500 * eatAdot
                    elif (manhattanDistance(newPos, curGhostPos[i]) == 1):
                        score += 50 * eatAdot
                    else:
                        score += 50 / manhattanDistance(newPos, curGhostPos[i]) * eatAdot
            if (not powerflag):
                if (manhattanDistance(newPos, curGhostPos[i]) <= 1):
                    score -= 1000 * eatAdot
                elif (manhattanDistance(newPos, curGhostPos[i]) == 2):
                    score -= 100 * eatAdot
                else:
                    score -= 1 / manhattanDistance(newPos, curGhostPos[i]) * eatAdot
            i += 1

        # distance to food
        for food in curFoodList:
            if (manhattanDistance(newPos, food) == 0):
                score += eatAdot
            elif (manhattanDistance(newPos, food) == 1):
                score += 0.75 * eatAdot
            else:
                score += 1 / manhattanDistance(newPos, food) * eatAdot

        # distance to capture
        for curCapsule in curCapsules:
            if (manhattanDistance(newPos, curCapsule) == 0):
                score += 100 * eatAdot
            elif (manhattanDistance(newPos, curCapsule) == 1):
                score += 10 * eatAdot
            else:
                score += 10 / manhattanDistance(newPos, curCapsule) * eatAdot

        # --- specification handling --- #
        # penalize the stop action
        if (action == "Stop"):
            score *= 0.5
        
        # reward for ending the game for several foods left
        if (len(curFoodList) <= 2):
            for food in curFoodList:
                if (manhattanDistance(newPos, food) == 0):
                    score += 5 * eatAdot
                elif (manhattanDistance(newPos, food) == 1):
                    score += 0.75 * 5 * eatAdot
                else:
                    score += 5 / manhattanDistance(newPos, food) * eatAdot

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

import math

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def maxValue(self, gameState, depth, agnetIndex = 0):
        if (depth <= 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        
        v = - math.inf
        if (depth == self.depth):
            actionBest = None
            for action in gameState.getLegalActions(agnetIndex):
                temp = self.minValue(gameState.generateSuccessor(agnetIndex, action), 
                                     depth, 
                                     agnetIndex + 1)
                if (v < temp):
                    v = temp
                    actionBest = action
            return actionBest

        else:
            for action in gameState.getLegalActions(agnetIndex):
                temp = self.minValue(gameState.generateSuccessor(agnetIndex, action), 
                                     depth, 
                                     agnetIndex + 1)
                v = max(v, temp)
            return v

    def minValue(self, gameState, depth, agnetIndex):
        if (depth <= 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        
        v = math.inf
        for action in gameState.getLegalActions(agnetIndex):
            if (agnetIndex != gameState.getNumAgents() - 1):
                v = min(v, 
                        self.minValue(gameState.generateSuccessor(agnetIndex, action), 
                                      depth,
                                      agnetIndex + 1
                                      )
                    )
            else:
                v = min(v, 
                        self.maxValue(gameState.generateSuccessor(agnetIndex, action),
                                      depth - 1,
                                      self.index
                                      )
                    )
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, self.depth, self.index)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, gameState, depth, alpha, beta, agnetIndex = 0):
        if (depth <= 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        
        v = - math.inf
        if (depth == self.depth):
            actionBest = None
            for action in gameState.getLegalActions(agnetIndex):
                temp = self.minValue(gameState.generateSuccessor(agnetIndex, action), 
                                     depth, 
                                     max(alpha, v), beta,
                                     agnetIndex + 1)
                if (v < temp):
                    v = temp
                    actionBest = action
            
            # alpha = max(alpha, v)
            return actionBest

        else:
            for action in gameState.getLegalActions(agnetIndex):
                temp = self.minValue(gameState.generateSuccessor(agnetIndex, action), 
                                     depth, 
                                     max(alpha, v), beta,
                                     agnetIndex + 1)
                v = max(v, temp)
                if (v >= beta):
                    return v
                    
            # alpha = max(alpha, v)
            return v

    def minValue(self, gameState, depth, alpha, beta, agnetIndex):
        if (depth <= 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        
        v = math.inf
        for action in gameState.getLegalActions(agnetIndex):
            if (agnetIndex != gameState.getNumAgents() - 1):
                v = min(v, 
                        self.minValue(gameState.generateSuccessor(agnetIndex, action), 
                                      depth,
                                      alpha, min(beta, v),
                                      agnetIndex + 1
                                      )
                    )
                if (v <= alpha):
                    return v
            else:
                v = min(v, 
                        self.maxValue(gameState.generateSuccessor(agnetIndex, action),
                                      depth - 1,
                                      alpha, min(beta, v), 
                                      self.index
                                      )
                    )
                if (v <= alpha):
                    return v
        
        # beta = min(beta, v)
        return v
    
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, self.depth, -math.inf, math.inf, self.index)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxValue(self, gameState, depth, agnetIndex = 0):
        if (depth <= 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        
        v = - math.inf
        if (depth == self.depth):
            actionBest = None
            for action in gameState.getLegalActions(agnetIndex):
                temp = self.chanceValue(gameState.generateSuccessor(agnetIndex, action), 
                                        depth, 
                                        agnetIndex + 1)
                if (v < temp):
                    v = temp
                    actionBest = action
            return actionBest

        else:
            for action in gameState.getLegalActions(agnetIndex):
                temp = self.minValue(gameState.generateSuccessor(agnetIndex, action), 
                                     depth, 
                                     agnetIndex + 1)
                v = max(v, temp)
            return v

    def chanceValue(self, gameState, depth, agnetIndex):
        if (depth <= 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        
        actions = gameState.getLegalActions(agnetIndex)
        n = len(actions)
        v = 0
        for action in actions:
            if (agnetIndex != gameState.getNumAgents() - 1):
                v += 1 / n * self.chanceValue(gameState.generateSuccessor(agnetIndex, action), 
                                              depth,
                                              agnetIndex + 1)
            else:
                v += 1 / n * self.chanceValue(gameState.generateSuccessor(agnetIndex, action),
                                              depth - 1,
                                              self.index)
        return v
    
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, self.depth, self.index)
        

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # --- info of current state --- #
    curPos = currentGameState.getPacmanPosition() # pacman position
    curFood = currentGameState.getFood() # remaining food
    curFoodList = curFood.asList() # food position list
    curCapsules = currentGameState.getCapsules() # captures postion list
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates] 
    curGhostPos = currentGameState.getGhostPositions() # ghost position

    score = 0 # scoring indicator
    eatAdot = 10 # reward unit
    powerFlags = [] # whether if you eat a capture

    for curScaredTime in curScaredTimes:
        if (curScaredTime != 0):
            powerFlags.append(True)
        else:
            powerFlags.append(False)

    # distance to ghost
    i = 0
    for powerflag in powerFlags:
        # once eat a capture, try to eat ghost~
        if (powerflag):
            if (manhattanDistance(curPos, curGhostPos[i]) < curScaredTimes[i]):
                if (manhattanDistance(curPos, curGhostPos[i]) <= 1):
                    score += 500 * eatAdot
                elif (manhattanDistance(curPos, curGhostPos[i]) == 2):
                    score += 50 * eatAdot
                else:
                    score += 50 / manhattanDistance(curPos, curGhostPos[i]) * eatAdot
        if (not powerflag):
            if (manhattanDistance(curPos, curGhostPos[i]) <= 1):
                score -= 1000 * eatAdot
            elif (manhattanDistance(curPos, curGhostPos[i]) == 2):
                score -= 10 * eatAdot
            else:
                score -= 1 / manhattanDistance(curPos, curGhostPos[i]) * eatAdot
        i += 1

    # distance to food
    score -= eatAdot * len(curFoodList) # penalize for more food it lefts
    for food in curFoodList:
        if (manhattanDistance(curPos, food) <= 1):
            score += eatAdot
        else:
            score += 1 / manhattanDistance(curPos, food) * eatAdot

    # distance to capture
    score -= 100 * eatAdot * len(curCapsules) # penalize for captures it lefts
    for curCapsule in curCapsules:
        if (manhattanDistance(curPos, curCapsule) <= 1):
            score += 100 * eatAdot
        elif (manhattanDistance(curPos, curCapsule) == 1):
            score += 10 * eatAdot
        else:
            score += 10 / manhattanDistance(curPos, curCapsule) * eatAdot

    # --- specification handling --- #
    
    # reward for ending the game for several foods left
    # if (len(curFoodList) <= 2):
    #     for food in curFoodList:
    #         if (manhattanDistance(curPos, food) == 0):
    #             score += 5 * eatAdot
    #         elif (manhattanDistance(curPos, food) == 1):
    #             score += 0.75 * 5 * eatAdot
    #         else:
    #             score += 5 / manhattanDistance(curPos, food) * eatAdot

    return score

# Abbreviation
better = betterEvaluationFunction
