import numpy as np
class NN:
    
    def __init__(self):
        #get input
        self.input= np.zeros(2278,1)
        self.output=np.zeros((4,1))
        self.layer1=np.zeros((569,1))
        #intializing w&b
        self.w1=np.random.rand(self.layer1.shape[0],self.input.shape[0])
        self.w2=np.random.rand(self.output.shape[0],self.layer1.shape[0])
        self.b1=np.random.rand(self.layer1.shape[0],1)
        self.b2=np.random.rand(self.output.shape[0],1)

    def sigmoid(self,x):
        for i in range(x.itemsize):
            for j in range(x.size/x.itemsize):
                x[i][j] = 1/(1+np.exp(-x[i][j]))
        return x
    
    def sigmoid_derivative(self, x):
        for i in range(x.itemsize):
            for j in range(x.size/x.itemsize):
                x[i][j] = x[i][j] * (1 - x[i][j])
        return x

    def feedforward(self):
        self.layer1 = np.dot(self.input, self.w1))+self.b1
        self.output = np.dot(self.input,self.w2))+self.b2
        #after applying activation function
        self.activelayer1 = sigmoid(self.layer1)
        self.activeoutput = sigmoid(self.output)
        
    def setweight_bais(self,w1,w2,b1,b2):
        self.w1=w1
        self.w2=w2
        self.b1=b1
        self.b2=b1

    def predict(self,inputVal):
        self.input = inputVal
        feedforward()
        return self.output
    
    def backpropagation(self,target):
        #y! = X((w2* X((w1*input)+b1)) + b2)
         x= np.subtract(self.activeoutput ,target)
         x = np.multiply(x,2)
         b2_derivative = np.dot(x,sigmoid_derivative(self.output))
         w2_derivative = np.dot(b2_derivative, self.activelayer1)
         y = np.dot(b2_derivative,self.w2)
         b1_derivative = np.dot(y, sigmoid_derivative(self.layer1))
         w1_derivative = np.dot(b1_derivative, self.input)

         #update weight and bais
         self.w1= self.w1 - 0.1 * w1_derivative
         self.w2= self.w2 - 0.1 * w2_derivative
         self.b1= self.b1 - 0.1 * b1_derivative
         self.b2= self.b2 - 0.1 * b2_derivative
        
    def train(self,inputVal,target,iteration):
        #find cost and error rate
        self.input = inputVal
        for i  in interaation: 
            feedforward()
            print ("error rate ")
            print (avg((self.output - target)**2))
            backpropagation(target)

    def getweight_bais(self):
        return [self.w1,self.b1,self.w2,self.b2]
                
        
