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


from util import manhattanDistance, Counter
from game import Directions
import random, util
import math
import time
from game import Agent
from ghostAgents import DirectionalGhost
from util import manhattanDistance, Counter
from game import Directions
import random, util
import math
import time
from game import Agent
from ghostAgents import DirectionalGhost
from ghostAgents import RandomGhost
from featureExtractors import SimpleExtractor 
from util import Stack,Queue,PriorityQueue
from mcts import Node                                           


from game import Agent
old_move = (0,0)
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
        global old_move
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        successorGameState = gameState.generatePacmanSuccessor(legalMoves[chosenIndex])
        newPos = successorGameState.getPacmanPosition()
        if min(scores) > -501 :
          if old_move[0] == newPos[0] and old_move[1] == newPos[1]:
            # dummy = scores
            # dummy.sort(reverse=True)
            # best = dummy[0]
            # for i in dummy:
            #   if i < best:
            #     bestIndices = [index for index in range(len(scores)) if scores[index] == i]
            #     chosenIndex = random.choice(bestIndices)
            old_move = gameState.getPacmanPosition()
            return  random.choice(legalMoves)
        "Add more of your code here if you want to"
        old_move = gameState.getPacmanPosition()
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()            
        currentfood = currentGameState.getFood()                  
        newFood = successorGameState.getFood()                            
        newGhostStates = successorGameState.getGhostStates() 
        width,height = len(newFood[:]),len(newFood[0][:])   
        escape_ghost,eat_ghost,eat_food,eat_cap  = -2500,5000,200,500                                            
        if action == "Stop":
           return -500
        if successorGameState.isWin(): 
          return  1000
        if  successorGameState.isLose():
          return -1000              
        points_ghost = 0.0                                       
        for ghost in newGhostStates:                               
          d = manhattanDistance(newPos, ghost.getPosition())
          if ghost.scaredTimer > 0:                      
            if d == 0: points_ghost += eat_ghost       
            elif d < 5: 
              points_ghost += (eat_ghost/d)   
          else:
            if d < 2: 
              points_ghost += (escape_ghost)   
            if d == 3 : 
              points_ghost += (escape_ghost/3)   
        points_food = 0.0                                        
        for x in range(width):                                     
          for y in range(height):                                  
            if(currentfood[x][y]):                             
              d = manhattanDistance(newPos, (x,y))          
              if(d == 0): 
                points_food += eat_food
              elif d<=3:
                points_food += 1      
              else: 
                points_food += 0.1/((d))             
        points_cap = 0.0                                       
        for cap in currentGameState.getCapsules():            
          d = manhattanDistance(newPos, cap)                
          if(d == 0): 
            points_cap += eat_cap             
          else: 
            points_cap += 10.0/d                                                          
        return points_ghost * 1 + points_food * (10 if points_ghost>=0 else 0.1) + points_cap * (1 if points_ghost>=0 else 0.1)

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

    number_of_nodes = []
    depth_of_tree = []
    time_per_moves = []                                          
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '4'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
class MinimaxAgent(MultiAgentSearchAgent):
    
    def minimax(self, gameState, depth, agent_index):                                                                                                               
      if gameState.isWin() or gameState.isLose(): 
        return self.evaluationFunction(gameState)   
      finalaction = 'null' 
      if agent_index == 0:                                                                     
        score = float('-inf')                                                                        
        legalMoves = gameState.getLegalActions(agent_index)
        i = 0
        while i < len(legalMoves):                                                                                                    
            next = gameState.generateSuccessor(agent_index, legalMoves[i])                              
            points = self.minimax(next, depth, agent_index + 1)
            if (score < points):                                                                      
              score, finalaction = points, legalMoves[i]
            i+=1                                                             
        if depth == 1: 
          return finalaction

      else:                                                                                   
        score = float('inf')                                                                        
        legalMoves =  gameState.getLegalActions(agent_index)
        agents = gameState.getNumAgents()
        i = 0
        while i < len(legalMoves):                                                               
            next = gameState.generateSuccessor(agent_index, legalMoves[i])                              
            if agent_index == agents - 1:                                                     
              if depth == self.depth: 
                points = self.evaluationFunction(next)                         
              else: 
                points = self.minimax(next, depth+1, 0)                                           
            else: 
              points = self.minimax(next, depth, agent_index+1)                                   
            if score > points: 
              score, finalaction = points, legalMoves[i]
            i+=1 

      return score                                                                             
                        
    def getAction(self, gameState):
      """ Returns the minimax action from the current gameState using self.depth and self.evaluationFunction. """
      return self.minimax(gameState, 1, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    
    def alphabetapruning(self, gameState, depth, agent_index,alpha,beta):
      if gameState.isWin() or gameState.isLose(): 
        return self.evaluationFunction(gameState)   
      finalaction = 'null'
      if agent_index == 0:                                                                     
        score = float('-inf')                                                                     
        legalMoves = gameState.getLegalActions(agent_index)                                     
        for action in legalMoves:                                                              
            next = gameState.generateSuccessor(agent_index, action)                              
            points = self.alphabetapruning(next, depth, agent_index + 1, alpha,beta)                              
            if points > beta: 
              return points                                                             
            if points > score:  
              score, finalaction = points, action
            alpha = max(score, alpha)
        if (depth == 1): 
          return finalaction

      else:
        score = float('inf')
        legalMoves =  gameState.getLegalActions(agent_index)
        agents = gameState.getNumAgents()
        for action in legalMoves:
            next = gameState.generateSuccessor(agent_index, action)
            if agent_index == agents - 1:
              if depth == self.depth: 
                points = self.evaluationFunction(next)
              else: 
                points = self.alphabetapruning(next, depth+1, 0,alpha,beta)
            else: 
              points = self.alphabetapruning(next, depth, agent_index+1,alpha,beta)
            if points < alpha: 
              return points
            if score > points: 
              score, finalaction = points, action
            beta = min(score, beta)
      return score

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        return self.alphabetapruning(gameState, 1, 0, float('-inf'),float('inf'))
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState, depth, agent_index):
      
      if gameState.isWin() or gameState.isLose():
        result = self.evaluationFunction(gameState)
        return (result, '')
      elif (depth == 0) :
        result = self.evaluationFunction(gameState)
        return (result, '')
      else:
        y = ''
        if agent_index == gameState.getNumAgents() - 1: 
          depth = depth - 1  
        if agent_index == 0: 
          x = float('-inf')
        else: 
          x = 0
        index = (agent_index + 1) % gameState.getNumAgents()
        legalMoves = gameState.getLegalActions(agent_index)
        i = 0
        while i < len(legalMoves):
          next =gameState.generateSuccessor(agent_index, legalMoves[i]) 
          value = self.expectimax(next, depth, index)
          if agent_index == 0:
            if  value[0] > x:
              y = legalMoves[i]              
              x = value[0]
          else:
            y = legalMoves[i]
            x += (1.0/len(legalMoves))*value[0]
          i+=1
        return (x, y)  
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax(gameState, self.depth, 0)[1]




class MCTS_Agent(MultiAgentSearchAgent):
    Node_Count = 0
    c_tree = None

    def __init__(self, steps='500', simDepth='10', exploreVar=250,
                     tillBored='80', optimism='0.2'):
                 
        self.iterations = int(steps)  
        self.simpulation_depth = int(simDepth)
        self.tillbored_steps = int(tillBored)
        self.e_algo_var = float(exploreVar)        
        self.feat_extractor = SimpleExtractor()
        self.choose_algo = 'best_combination'
        self.Ghost_randomness = float(optimism)
        self.weights = Counter({'closest-food': -3.01, 'bias': 211.11, '#-of-ghosts-1-step-away': -113.99, 'eats-food': 269.99})

    def getAction(self, gameState):
        
            
        def Pacman_action(state):
            Possible_moves = state.getLegalActions(0)
            length = len(Possible_moves)
            if length > 0:
                best_action = [Possible_moves[0]]
                best_value = self.weights*self.feat_extractor.getFeatures(state,Possible_moves[0])
                for move_current in Possible_moves[1:]:
                    Current_value = self.weights*self.feat_extractor.getFeatures(state,move_current)
                    if Current_value > best_value :
                        best_action = [move_current]
                        best_value = Current_value
                    elif best_value == Current_value:
                      best_action.append(move_current)
                select_act = random.choice(best_action)
                new_state = state.generateSuccessor(0, select_act)
                return new_state, select_act
            else: return None

         
        def Selection_MCTS(tree):
            if not tree.children: return tree
            b_child = tree.upperconfidencebound(self.e_algo_var)
            return Selection_MCTS(b_child)

        def Expansion_MCTS(leaf):
            leaf.generate_children()
            self.Node_Count += len(leaf.children)
                
        def Back_propagation_MCTS(result, node):
            win, score = result
            #update the score
            node.score_update(win, score)
            if node.parent is None: return
            Back_propagation_MCTS(result, node.parent)

        def heuristic_func(state):
            score = 0
            if state.isWin():
              return 1,10000
            if state.isLose():
              return 0,-10000
            newPos = state.getPacmanPosition()                            
            newFood = state.getFood()                            
            newGhostStates = state.getGhostStates()
            closest_distance = float('inf')
            score = state.getScore()
            for closest_food in newFood.asList():
                di = manhattanDistance(newPos, closest_food)
                if closest_distance > di:
                  closest_distance = di
          
            # score += state.getScore()+400 - closest_food * (0.5 if len(newFood.asList())>=5 else 2.5) -0.1*len(newFood.asList())-  0.5*len(newFood.asList()) - len(state.getCapsules())
            # score = score + state.getScore()*(1 if len(newFood.asList())>=10 else 2)+(800 if len(newFood.asList())>=10 else 400)- closest_distance * (0.5 if len(newFood.asList())>=5 else closest_distance^10) -  len(newFood.asList())*(0.25 if len(newFood.asList())>=10 else 10) 
            # score = score + state.getScore()*(1 if len(newFood.asList())>=10 else 2)+(800 if len(newFood.asList())>=10 else 400) - closest_distance * (0.5 if len(newFood.asList())>=10 else closest_distance) -  len(newFood.asList())*(0.25 if len(newFood.asList())>=10 else 5)
            score =  score + 800 - closest_distance * 0.5
            # print(score)
            return 0.5 if score >0 else 0, score

            
        def Tree_maximum_depth(node, c_depth=0):
            check = c_depth
            length = len(node.children)
            if(length > 0):
                check = Tree_maximum_depth(node.children[0], check + 1)
                i = 1
                while(i < length):
                  dc = Tree_maximum_depth(node.children[i], check + 1)
                  if(dc > check):
                    check = dc
                  i+=1
            return check


        def Simulation_MCTS(node, agent_index=0, Heuristic_MCTS=heuristic_func):
                state = node.state
                if random.random() < self.Ghost_randomness:
                  ghosts = [RandomGhost(i+1) for i in range(state.getNumAgents())]
                else: ghosts = [DirectionalGhost(i+1) for i in range(state.getNumAgents())]
                for _ in range(self.simpulation_depth):
                    while agent_index < state.getNumAgents():
                        if state.isWin(): return state.isWin(), state.getScore()+400
                        if state.isLose(): return state.isWin(), state.getScore()-400
                        if agent_index == 0: state, action = Pacman_action(state)
                        # if agent_index == 0:
                        #     x = ReflexAgent()
                        #     state = state.generateSuccessor(0, x.getAction(state))
                        else:
                            ghost = ghosts[agent_index-1]
                            state = state.generateSuccessor(agent_index, ghost.getAction(state))
                        agent_index += 1
                    agent_index = 0
                return Heuristic_MCTS(state)
            
        def BFS(c_node, target_search, depth=0):
            
            state_fnd = None
            fringe = Queue()
            fringe.push(c_node)
            while not fringe.isEmpty():
                s= fringe.pop()
                if target_search == s.state:
                    state_fnd = s   
                    return state_fnd
                    break
                for c_child in s.children:
                        fringe.push(c_child)
            return state_fnd
        

        # Monte Carlo tree search  
        
        start_time = time.time()
        # filename = "tree.pkl"
        # if os.path.isfile(filename):
        #     t = pickle.load(open(filename, "rb"))
        #     tree = BFS(t, gameState, 0)
        #     print("found the pickle file")
        #     if tree is None:
        #         print(t.id_node,"trying with the MCTS_Agent")
        #         tree = BFS(MCTS_Agent.c_tree, gameState, 0)
        #         print("MCTS_Agent result: ",tree)
        #     print(t.id_node,t.children,tree)
        #     os.remove(filename)
        # else:
        if MCTS_Agent.c_tree is not None:
            tree = BFS(MCTS_Agent.c_tree, gameState, 0)
        else:
            tree = None
        if tree is None:
           
            tree = Node(gameState, action=None, parent=None)
            
        else:
            tree.parent = None
        # tree = Node(gameState, action=None, parent=None)

        Iteration_counter = 0
        Bored_Steps = 0
        Previous_Action = -1
        
        while Iteration_counter < self.iterations:
            leaf = Selection_MCTS(tree)
            if self.e_algo_var -0.5 > 50:
                self.e_algo_var -= 0.5
            # print(leaf.state)
            Expansion_MCTS(leaf)
            if leaf.children:
                child = random.choice(leaf.children)
                result = Simulation_MCTS(child, child.agent_index+1)
                Back_propagation_MCTS(result, child)
            else:
                result = leaf.state.isWin(), leaf.state.getScore()
                Back_propagation_MCTS(result, leaf)
                # if leaf.state.isWin() or leaf.state.isLose():
                #     pickle.dump(MCTS_Agent.c_tree,open(filename,"wb"))
                
                break
            Iteration_counter +=1
            Current_Action = tree.Selection_strategy()
            if Current_Action == Previous_Action:
                Bored_Steps += 1
                if self.e_algo_var  < 100:
                    self.e_algo_var = 150
                if Bored_Steps >= self.tillbored_steps: break
            else:
                Previous_Action = Current_Action
                Bored_Steps = 0
            
        
        Node.id_node = 0
        MCTS_Agent.c_tree = tree
        # pickle.dump(MCTS_Agent.c_tree,open(filename,"wb"))
        act = tree.Selection_strategy()
        # print(act,Tree_maximum_depth(tree), self.Node_Count)
        #end-time
        end_time = time.time()
        # self.e_algo_var = 150
        
        MultiAgentSearchAgent.depth_of_tree.append(Tree_maximum_depth(tree))
        MultiAgentSearchAgent.number_of_nodes.append(self.Node_Count)
        MultiAgentSearchAgent.time_per_moves.append(end_time - start_time)
        self.Node_Count = 0
        # dummy = tree.state.generateSuccessor(0, act)
        
        #return the relevant action
        # print(sum(MultiAgentSearchAgent.time_per_moves))
        
        
        
        return act    