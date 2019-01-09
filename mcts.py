# This module will implement a simple monte carlo tree search
# algorithm. We will test it on tic tac toe, connect 4
# and the traveling salesman problem. States will be
# represented as vectors.

import numpy as np
import math as m
import copy
import random
random.seed()
class Node:
   # Each of the nodes in our monte carlo tree will be
   # one of these nodes. For the time being we will
   # create a node always passing in the corresponding
   # state as the parameter.
   def __init__(*arg):
      self = arg[0];
      self.s = arg[1];
      self.n = 0;
      self.w = 0;
      # We will have a pointer to the parent
      self.p = None;
      # we will have a dictionary of pointers to the children
      self.c = {};
      # t stands for taste and will be used for multiplayer
      # games. 0 is o's and 1 is x's we want x to go first
      # so we set the 0th move to o's
      self.t = 0;
      #print len(self.c);

class MCTS:
   # This class will implement the actual algorithms
   # to initialize tree we create the root node and
   # pass in the state vector for the root node.
   def __init__(*arg):
      self = arg[0];
      self.r = Node(arg[1]);
      # k is the constant that is used to create the balance
      # between exploration and specialization.
      self.k = 1

   # This function selects and returns the node that is best
   # candidate base off w/n + k*sqrt(2ln(n_p)/ln(n_c)) 
   def select(self,theNode,tasteFunc):
      scores = np.zeros(len(theNode.c))
      #print len(scores);
      for i in range(0,len(scores)):
         #print "Inside select for loop";
         # if any of the nodes where never visited return
         # that node emidiately.
         if (theNode.c[i].n == 0):
            return theNode.c[i]
         else :
            tempNum = tasteFunc((theNode.c[i].w + .0)/theNode.c[i].n,theNode.t);
            scores[i] = tempNum + self.k*m.sqrt(2*m.log(theNode.n)/(.0+theNode.c[i].n));
            #print scores;
            #print np.argmax(scores);
      return theNode.c[int(np.argmax(scores))]

   # This function applies select in a while loop until we
   # have travesed the tree to get to a leaf assuming two 
   # players
   def nextLeaf2p(self,childrenFunc):
      currentNode = self.r;
      while (len(currentNode.c) > 0):
         currentNode = self.select(currentNode,self.tFunc1);
      currentNode.c = childrenFunc(currentNode);
      return currentNode;

   # This function defines how to process the w/n term for
   # different types of games, in this case, for 2 player
   # games.
   def tFunc1(self,rate,taste):
      if taste == 0:
         return rate;
      if  taste == 1:
         return 1 - rate;

   # This function requires a function pointer that takes
   # a current state and returns a possible next state 
   # choosen at random.  
   def nextRandNode(self,theNode,randNodeFunc):
      return randNodeFunc(theNode);

   # This function requires a function pointer and takes in
   # a current node and returns the dictionary of children
   # nodes and assigns them to the c member of the current
   # node
   def createChildren(self,theNode,childrenFunc):
      theNode.c = childrenFunc(theNode);

   # This is our simulate function. We will do something a 
   # little different here in that we will accumulate the
   # score. In the future, I want to run this code in a way
   # that allows us to reward a player for surviving. This
   # accumulator will have no effect if all except the final
   # state have zero score. So accumulating scores can be
   # used in both modalities.
   def simmulate(self, startNode, randNodeFunc, scoreFunc):
      # Note to self, we need to be carefull of copying
      # data versus copying pointers.
      cNode = startNode;
      score = 0;
      while (cNode != None):
         score += scoreFunc(cNode)
         cNode = self.nextRandNode(cNode,randNodeFunc)
      pNode =startNode
      while (pNode != None):
         pNode.w += score
         pNode.n += 1
         pNode = pNode.p;

   def runMCTS(self,iterNum,rnFunc,cFunc,sFunc):
      for i in range(0,iterNum):
         # find our next leaf
         leafNode = self.nextLeaf2p(cFunc);
         if (leafNode):
            self.simmulate(leafNode,rnFunc,sFunc);

# This is our function to tell if the game is over
# and if so, if it is won lost or tied for tic tac
# toe. Returns [isOver,score]. We assume x wins gives
# a score of 1, o wins gives a score of 0, and tie 
# gives a score of .5
def tictactoeOver(theNode):
   isOver = 0;
   score = 0;
   if (theNode.s[0] == theNode.s[1] == theNode.s[2] == -1):
      isOver = 1;
      score = 0;   
   if (theNode.s[0] == theNode.s[1] == theNode.s[2] == 1):
      isOver = 1;
      score = 1;
   if (theNode.s[3] == theNode.s[4] == theNode.s[5] == -1):
      isOver = 1;
      score = 0;   
   if (theNode.s[3] == theNode.s[4] == theNode.s[5] == 1):
      isOver = 1;
      score = 1;
   if (theNode.s[6] == theNode.s[7] == theNode.s[8] == -1):
      isOver = 1;
      score = 0;   
   if (theNode.s[6] == theNode.s[7] == theNode.s[8] == 1):
      isOver = 1;
      score = 1;
   if (theNode.s[0] == theNode.s[3] == theNode.s[6] == -1):
      isOver = 1;
      score = 0;   
   if (theNode.s[0] == theNode.s[3] == theNode.s[6] == 1):
      isOver = 1;
      score = 1;
   if (theNode.s[1] == theNode.s[4] == theNode.s[7] == -1):
      isOver = 1;
      score = 0;   
   if (theNode.s[1] == theNode.s[4] == theNode.s[7] == 1):
      isOver = 1;
      score = 1;
   if (theNode.s[2] == theNode.s[5] == theNode.s[8] == -1):
      isOver = 1;
      score = 0;   
   if (theNode.s[2] == theNode.s[5] == theNode.s[8] == 1):
      isOver = 1;
      score = 1;
   if (theNode.s[0] == theNode.s[4] == theNode.s[8] == -1):
      isOver = 1;
      score = 0;   
   if (theNode.s[0] == theNode.s[4] == theNode.s[8] == 1):
      isOver = 1;
      score = 1;
   if (theNode.s[2] == theNode.s[4] == theNode.s[6] == -1):
      isOver = 1;
      score = 0;   
   if (theNode.s[2] == theNode.s[4] == theNode.s[6] == 1):
      isOver = 1;
      score = 1;
   if (0 not in theNode.s):
      isOver = 1;
      score = .5;
   #print theNode.s;
   return [isOver,score]

# Now, we just need to define our auxilary functions.  
# these functions are going to depend on which game 
# we are playing. A Tic Tac Toe board will be a 9
# dim vector of 0's 1's and -1's where the 0's are
# empty spots, 1's are x's and -1's are o's
def tictactoeChildren(theNode):
   [isOver, score] = tictactoeOver(theNode);
   if isOver is not 0:
      return None;
   else :
      children = {};
      i = 0;
      for j in range(0,9):
         if theNode.s[j] == 0:
            temp = copy.copy(theNode.s)
            # temp[j] is set to 1 if its x's turn 
            # and 0 if its o's turn. The t member
            # stores the last turn taken.
            temp[j] = 1 - 2*theNode.t
            children[i] = Node(temp);
            children[i].t = 1 - theNode.t
            children[i].p = theNode;
            # increment i
            i += 1
      return children;

# This function returns a child node at random.
def tictactoeRandChild(theNode):
   [isOver, score] = tictactoeOver(theNode);
   if isOver is not 0:
      return None;
   else :
      count = 0
      for j in range(0,9):
         if theNode.s[j] == 0:
            count += 1;
      randInt = random.randint(1,count);
      newCount = 0
      for j in range(0,9):
         if theNode.s[j] == 0:
            newCount += 1
         if randInt == newCount:
            temp = copy.copy(theNode.s)
            temp[j] = 1 - 2*theNode.t;
            child = Node(temp);
            child.t = 1 - theNode.t;
            child.p = theNode;
            return child;
   # This return statement should never be reached
   return None;

# This function is used to evaluate the score of the current
# state. In tic tac toe, score is zero until the end state
# is reached at which point the score may be 0, 1 or .5
def tictactoeScore(theNode):
   [isOver, score] = tictactoeOver(theNode);
   return score;
      
