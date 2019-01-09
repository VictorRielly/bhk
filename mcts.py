# This module will implement a simple monte carlo tree search
# algorithm. We will test it on tic tac toe, connect 4
# and the traveling salesman problem. States will be
# represented as vectors.

import numpy as np
import math as m
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
      # games.
      self.t = 0;
      print len(self.c);

class MCTS:
   # This class will implement the actual algorithms
   # to initialize tree we create the root node and
   # pass in the state vector for the root node.
   def __init__(*arg):
      self = arg[0];
      self.r = Node(arg[1]);
      self.r.n=1
      # k is the constant that is used to create the balance
      # between exploration and specialization.
      self.k = 1

   # This function selects and returns the node that is best
   # candidate base off w/n + k*sqrt(2ln(n_p)/ln(n_c)) 
   def select(self,theNode,tasteFunc):
      scores = np.zeros(len(theNode.c))
      for i in range(0,len(scores)):
         # if any of the nodes where never visited return
         # that node emidiately.
         if (theNode.c[i].n == 0):
            return theNode.c[i]
         else :
            scores[i] = tasteFunc((theNode.c[i].w + .0) / theNode.c[i].n,theNode.t) + self.k*m.sqrt(2*m.log(theNode.n)/(.0+theNode.c[i].n));
      return theNode.c[np.argmax(scores)]

   # This function applies select in a while loop until we
   # have travesed the tree to get to a leaf assuming two 
   # players
   def nextLeaf2p(self):
      currentNode = self.r;
      while (len(currentNode.c) > 0):
         currentNode = self.select(currentNode,self.tFunc1);

   # This function defines how to process the w/n term for
   # different types of games, in this case, for 2 player
   # games.
   def tFunc1(wrate,taste):
      if taste == 0:
         return wrate;
      if  taste == 1:
         return 1 - wrate;

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
      pNode = startNode
      while (pNode != None):
         pNode.w += score
         pNode.n += 1
         pNode = pNode.p;

   def runMCTS(self,iterNum,rnFunc,cFunc,sFunc):
      for i in range(0,iterNum):
         # find our next leaf
         leafNode = self.nextLeaf2p(self.r);
         self.simmulate(leafNode,rnFunc,scoreFunc);

# Now, we just need to define our auxilary functions.  


