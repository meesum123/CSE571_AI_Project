
from util import manhattanDistance, Counter
from game import Directions
import random, util
import math
import time
from game import Agent
from ghostAgents import DirectionalGhost
from ghostAgents import RandomGhost
from featureExtractors import SimpleExtractor         


class Node():
    """stores nodes in the search tree."""
    id_node = 0
    
    def __init__(self, state, action, parent, agent_index=0):

        self.state = state
        self.agent_index = agent_index
        self.parent = parent
        self.action = action
        self.wins_num = 0
        self.explored_num = 0 
        self.sum_score = 0 
        self.id_node = Node.id_node 
        Node.id_node += 1
        self.children = []
         
       
    def Selection_strategy(self):
        max_index = []
        best_score = -float('inf')
        for c_child in self.children:
            if c_child.explored_num:
                winaverage = c_child.wins_num / c_child.explored_num
                if winaverage > 0.1: c_score = (winaverage * c_child.sum_score) / c_child.explored_num
                else: c_score = winaverage
            else: c_score = -float('inf')
            if c_score > best_score:
                max_index = [c_child]
                best_score = c_score
            elif c_score == best_score:
                max_index.append(c_child)
        
        return random.choice(max_index).action
   
    # def find_action(self, b_child_algorithm='best_combination'):
        
    #     b_child = self.Selection_strategy()
       
    #     return b_child.action
    
    def upperconfidencebound(self, c=150.0):
        max_score = -99999999
        i = 0
        score_list = []
        while(i < len(self.children)):
            score = 0
            temp = self.children[i]
            if(temp.explored_num != 0):
                score = temp.sum_score/temp.explored_num
                score += c * (math.log(self.explored_num)/(temp.explored_num))
            else:
                score = 999999999
            score_list.append(score)
            if(score > max_score):
                max_score = score
            i+=1
        indices_high = []
        for i in range(len(score_list)):
            if(score_list[i] == max_score):
                indices_high.append(i)        
        return self.children[random.choice(indices_high)]
  
    # def exploit_explore(self, e_algorithm='ucb', e_variable='150'):
    #     #can remove e variable
    #     if e_algorithm == 'ucb':
    #         return self.upperconfidencebound(float(e_variable))
    #     else:
    #         return self.epsilongreed(float(e_variable))
    
    def generate_children(self):
        k = self.agent_index
        actions = self.state.getLegalActions(k)
        i = 0
        while(i < len(actions)):
            temp = Node(self.state.generateSuccessor(k, actions[i]), actions[i], parent = self, agent_index = (k + 1) % self.state.getNumAgents())
            self.children.append(temp)
            i += 1

    def score_update(self, win, score):
        self.explored_num +=1
        if self.agent_index != 1:
            self.wins_num -= float(not win)
            self.sum_score -= score
        else:
            self.wins_num +=  float(win)
            self.sum_score += score
            