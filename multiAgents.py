# multiAgents.py
# --------------
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #처음 스코어를 0으로 두고 시작.
        myscore = 0
        #음식을 먹는 것이 주 목적 이기 때무에, 음식을 먹으러 간다면, 10점을 준다.
        if currentGameState.getFood()[newPos[0]][newPos[1]]:
            myscore+=100
        #새로운 음식들에 대한 리스트를 받고,
        newFood = newFood.asList()
        min_food_distance = 100
        #음식 리스트에 대해서 하나씩 manhattanDistance를 구해서, 가장 가까운 위치를 찾아낸다.
        for onefood in newFood:
            food_distance = util.manhattanDistance(newPos, onefood)
            if min_food_distance >= food_distance:
                min_food_distance = food_distance

        #고스트 위치를 찾는다. 기본값을 100으로 설정한다.
        min_ghost_distance = 100
        #고스트 중 manhattanDistance가 가장 가까운 고스트를 찾아낸다.
        for one_ghost in successorGameState.getGhostPositions():
            ghost_distance = util.manhattanDistance(newPos, one_ghost)
            if ghost_distance<min_ghost_distance:
                min_ghost_distance=ghost_distance
        #만약 가장 가까운 고스트가 거리 3 이내에 있다면, 스코어에서 150점을 깎는다.
        if min_ghost_distance < 3 :
            myscore -= 150
        #food_distance는 작을수록 좋고, ghost_distance는 클수록 좋기때문에 아래와 같은 식을 적용하였다.
        return myscore + (5.0/min_food_distance) + (min_ghost_distance*0.01)

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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
        #현재 가능한 action들을 받는다.
        actions = gameState.getLegalActions(0)
        #maxgrade로 받을 값을 최저로 잡아둔다.
        maxscore = -2147000000
        #return할 action값을 기본으로 설정해둔다.
        action = Directions.EAST

        #이제 actions에 대한 for 문을 돌리면서 취해야할 action을 정한다.
        for i in actions:
            #하나의 action에 대해 score를 구한다.
            #우선, 함수 Min_Value를 호출해서 시작한다.
            action_score = self.Minimax_Min(gameState.generateSuccessor(0, i), 1, 0)

            #만약 이 action score를 받은것이 원래 가지고 있던 maxscore보다 크다면, maxscore와 action을 바꾼다.
            if (action_score) > maxscore:
                #action을 현재 for문에서 돌고있는 action으로 한다.
                 action = i
                #maxscore를 현재 score로 바꾼다.
                 maxscore = action_score
        #최종적으로 정해진 action을 선택한다.
        return action

    #Minimax_Min함수이다. Minimax_Max와 다른 점은, 유령은 갯수가 있기 때문에 agentIndex를 인자로 받는다.
    def Minimax_Min(self, gameState, agentIndex, depth):

        #현재의 agentIndex에서 가져올 수 있는 action의 수가 0개 일때, 점수를 반환한다.
        if len(gameState.getLegalActions(agentIndex)) == 0:
            return self.evaluationFunction(gameState)
        #만약 agent가 마지막 agent라면(유령 중 마지막 유령) 유령 agent 차례를 끝내고 Max로 넘어가서, pacman 차례로 돌린다.
        if agentIndex == gameState.getNumAgents()-1:
            #minlist라는 빈 리스트를 하나 생성한다.
            minlist = []
            #현재 agentIndex(유령)에서 취할 수 있는 action에 대해 for 문을 돌린다.
            for action in gameState.getLegalActions(agentIndex):
                #각각의 action에 대해 후계노드를 Minimax_Max값에 넣고 depth를 하나 추가한다. 그 값을 minlist에 추가한다.
                minlist.append(self.Minimax_Max(gameState.generateSuccessor(agentIndex, action),depth+1))
            #minlist값 중 최소값을 리턴한다.
            return min(minlist)
        #마지막 유령이 아니라면, 옆의 다른 유령으로 넘어간다.
        else:
            # minlist라는 빈 리스트를 하나 생성한다.
            minlist = []
            # 현재 agentIndex(유령)에서 취할 수 있는 action에 대해 for 문을 돌린다.
            for action in gameState.getLegalActions(agentIndex):
                # 각각의 action에 대해 후계노드를 Minimax_Min값에 넣고 agentIndex를 하나 추가한다. 그 값을 minlist에 추가한다.
                minlist.append(self.Minimax_Min(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth))
            # minlist값 중 최소값을 리턴한다.
            return min(minlist)
    #팩맨 차례에서 돌아가는 Minimax_Max함수이다.
    def Minimax_Max(self, gameState, depth):
        #만약 depth가 입력한 값과 같아진 경우, 점수를 반환한다.
        if depth == self.depth:
            return self.evaluationFunction(gameState)
        #만약 더이상 취할 수 있는 action이 없을 경우, 점수를 반환한다.
        if len(gameState.getLegalActions(0)) == 0:
            return self.evaluationFunction(gameState)
        #maxlist라는 빈리스트를 생성한다.
        maxlist=[]
        #현재 취할 수 있는 action에 대해 for문을 돌린다.
        for action in gameState.getLegalActions(0):
            #Minimax_Min 함수를 호출해서 돌리고, 각각의 값을 maxlist에 추가한다.
            maxlist.append(self.Minimax_Min(gameState.generateSuccessor(0, action), 1, depth))
        #maxlist중 가장 큰 값을 리턴한다.
        return max(maxlist)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        #alpha차례에서 돌아갈 함수이다.
        def alpha_function(alpha, beta, gameState, depth):
            #depth가 입력한 depth와 같아졌거나, 이기거나 지는 경우가 되면 점수를 반환한다.
            if depth==self.depth or gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState),None)
            #가능한 action들을 받는다.
            actions = gameState.getLegalActions(0)
            #가능한 action이 없을 때, 점수를 반환한다.
            if len(actions)==0:
                return (self.evaluationFunction(gameState),None)
            #for문을 돌기전에 v와 반환한 action을 설정한다.
            v = -100000
            action_real = None

            #actions를 돌면서 각 action에 대해 계산한다.
            for action in actions:
                #후속 agent를 설정한다. 이때, 유령으로 넘어가야하므로 beta_function으로 연결한다.
                next_value = beta_function(alpha, beta, gameState.generateSuccessor(0, action), depth, 1)[0]

                #이제 v를 최대값으로 바꿔주어야 한다.
                if next_value>v:
                    v = next_value
                    action_real = action
                #v가 beta보다 크다면, v와 그 action을 반환한다.
                if v>beta:
                    return (v, action_real)
                #alpha는 alpha와 v중 큰 값이다.
                alpha = max(alpha, v)
            return (v, action_real)

        #beta차례에서 돌아갈 함수이다.
        def beta_function(alpha, beta, gameState, depth, agent):
            # 가능한 action들을 받는다.
            actions = gameState.getLegalActions(agent)
            # 가능한 action이 없을 때, 점수를 반환한다.
            if len(actions) == 0:
                return (self.evaluationFunction(gameState), None)
            #v를 최대값으로 두고, action을 설정한다.
            v = 100000
            action_real = None

            #각 action에 대해서 돌려야 한다.
            for action in actions:
                #고스트의 경우 마지막 고스트인지 마지막 고스트가 아닌지에 대해 나누어야 한다.
                #마지막 고스트가 아닐 경우, 다른 고스트에게 넘겨야 한다.
                if agent < gameState.getNumAgents()-1:
                    next_value = beta_function(alpha, beta, gameState.generateSuccessor(agent, action), depth, agent+1)[0]
                #마지막 고스트라면 alpha_function으로 넘긴다.
                else:
                    next_value = alpha_function(alpha, beta, gameState.generateSuccessor(agent, action), depth+1)[0]

                #v와 next_value값을 비교한다.
                if next_value < v:
                    v = next_value
                    action_real = action
                #alpha와 v를 비교한다.
                if alpha>v:
                    return (v, action_real)
                #beta는 beta값과 v값중 작은 값이다.
                beta = min(beta, v)
            return (v, action_real)

        # 알파는 최소값, 베타는 최대값으로 잡는다.
        alpha = -100000
        beta = 100000
        #최종적으로 반환할 값은 alpha_function 쪽이다.
        return alpha_function(alpha,beta,gameState,0)[1]
       # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
