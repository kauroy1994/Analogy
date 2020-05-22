from random import random
from copy import deepcopy
from math import sqrt,exp

class Pairwise_transformer(object):

    def __init__(self,blocks = (1,1)):
        """constructor to initialize:
           1. number of transformer blocks
        """

        self.blocks = blocks
        self.delta = 0.001

    def sigmoid(self,x):
        """sigmoid function
           for output
        """

        return (exp(x)/float(1+exp(x)))

    def softmax(self,vector):
        """computes softmax of
           vector components
        """

        den = sum([exp(item) for item in vector])
        return ([exp(item)/float(den) for item in vector])

    def multiply(self,matrix,vector):
        """returns matrix vector
           product
        """

        product = []
        n = len(matrix)
        for i in range(n):
            item = sum([matrix[i][j]*vector[j] for j in range(n)])
            product.append(item)
        return (product)

    def add(self,vector1,vector2):
        """adds two vectors
        """

        n = len(vector1)
        return ([vector1[i]+vector2[i] for i in range(n)])

    def dot(self,vector1,vector2):
        """returns dot product
           of two vectors
        """
        n = len(vector1)
        return (sum([vector1[i]*vector2[i] for i in range(n)]))

    def scalar_multiply(self,scalar,vector):
        """scalar vector multiplication
        """

        return ([scalar*item for item in vector])

    def compute_value(self,values,attention):
        """computes linear combination
           for values using attention
           weights
        """

        n_values = len(values)
        value = [0.0 for i in range(len(values[0]))]
        for i in range(n_values):
            v = self.scalar_multiply(attention[i],values[i])
            value = self.add(value,v)

        return (value)

    def compute(self,ex1,ex2):
        """forward pass on both
           transformers
        """

        #==============FIRST TRANSFORMER====================
        block_input = deepcopy(ex1)
        for b in range(self.blocks[0]):
            next_block_input = []
            for hi in block_input:
                qi = self.multiply(self.q1_ws[b],hi)
                values = []
                attention = []
                for hj in block_input:
                    kj = self.multiply(self.k1_ws[b],hj)
                    vj = self.multiply(self.v1_ws[b],hj)
                    aij = self.dot(qi,kj)/float(sqrt(len(hi)*len(block_input)))
                    attention.append(aij)
                    values.append(vj)
                attention = self.softmax(attention)
                value = self.compute_value(values,attention)
                next_block_input.append(value)
            block_input = next_block_input

        #===========SECOND TRANSFORMER=====================
        block_input2 = deepcopy(ex2)
        for b in range(self.blocks[1]):
            next_block_input = []
            for hi in block_input2:
                qi = self.multiply(self.q2_ws[b],hi)
                values = []
                attention = []
                for hj in block_input2:
                    kj = self.multiply(self.k2_ws[b],hj)
                    vj = self.multiply(self.v2_ws[b],hj)
                    aij = self.dot(qi,kj)/float(sqrt(len(hi)*len(block_input2)))
                    attention.append(aij)
                    values.append(vj)
                attention = self.softmax(attention)
                value = self.compute_value(values,attention)
                next_block_input.append(value)
            block_input2 = next_block_input

        #==========DOT PRODUCT OF TWO TRANSFORMERS==========

        t1_output,t2_output = [],[]
        for hi1 in block_input:
            t1_output += hi1
        for hi2 in block_input2:
            t2_output += hi2
            
        return (self.sigmoid(self.dot(t1_output,t2_output)))
            

    def train(self,X1,X2):
        """transformer encoder trained
           using method specified
        """

        n = len(X1[0])
        n_points = len(X1)
        b1 = self.blocks[0]
        b2 = self.blocks[1]
        q1_ws,k1_ws,v1_ws = [],[],[]
        q2_ws,k2_ws,v2_ws = [],[],[]

        for b in range(b1):
            q1_ws.append([[random() for i in range(n)] for i in range(n)])
            k1_ws.append([[random() for i in range(n)] for i in range(n)])
            v1_ws.append([[random() for i in range(n)] for i in range(n)])

        for b in range(b2):
            q2_ws.append([[random() for i in range(n)] for i in range(n)])
            k2_ws.append([[random() for i in range(n)] for i in range(n)])
            v2_ws.append([[random() for i in range(n)] for i in range(n)])

        self.q1_ws = q1_ws
        self.k1_ws = k1_ws
        self.v1_ws = v1_ws
        self.q2_ws = q2_ws
        self.k2_ws = k2_ws
        self.v2_ws = v2_ws

        """ **TODO** THIS PART CONTINUE LATER
        
        for parameter in self.parameters:
            for i in range(n_points):
                for j in range(n_points):
                    output = self.compute(X1[i],X2[j])

        """
        

        

        

        

    
