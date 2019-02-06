import numpy as np
class NN:
    
    def __init__(self):
        #get input
        self.input= np.zeros((4,1)) #(2278,1)
        self.output=np.zeros((2,1)) #(4,1)
        self.layer1=np.zeros((3,1))   #(569,1)
        #intializing w&b
        self.w1=np.random.rand(self.layer1.shape[0],self.input.shape[0])
        self.w2=np.random.rand(self.output.shape[0],self.layer1.shape[0])
        self.b1=np.random.rand(self.layer1.shape[0],1)
        self.b2=np.random.rand(self.output.shape[0],1)

    def sigmoid(self,inputVal):
        sigmoid_list = [1 / float(1 + np.exp(- x)) for x in inputVal] 
##        for i in range(x.itemsize):
##            for j in range(x.size/x.itemsize):
##                x[i][j] = 1.0/(1.0+np.exp(-x[i][j]))
##                print ("{},{}->{}".format(i,j,x[i][j]))
        sigmoid_array = np.array([sigmoid_list])
        return sigmoid_array.T
    
    def sigmoid_derivative(self, x):
        for i in range(x.itemsize):
            for j in range(x.size/x.itemsize):
                x[i][j] = x[i][j] * (1 - x[i][j])
        return x

    def feedforward(self):
        self.layer1 = np.dot(self.w1, self.input)+self.b1
        #after applying activation function
        self.activelayer1 = self.sigmoid(self.layer1)
        
        self.output = np.dot(self.w2,self.activelayer1)+self.b2
        #after applying activation function
        self.activeoutput = self.sigmoid(self.output)
        
    def setweight_bais(self,w1,w2,b1,b2):
        self.w1=w1
        self.w2=w2
        self.b1=b1
        self.b2=b1

    def predict(self,inputVal):
        self.input = inputVal
        self.feedforward()
        return self.activeoutput
    
    def backpropagation(self,target):
        #y! = X((w2* X((w1*input)+b1)) + b2)
         x= np.subtract(self.activeoutput ,target)
         x = np.multiply(x,2)
         b2_derivative = np.dot(x.T,self.sigmoid_derivative(self.activeoutput)) #derivative of e is e
         w2_derivative = np.dot( self.activelayer1 , b2_derivative)
         y = np.dot(b2_derivative[0][0],self.w2)
         b1_derivative = np.dot(y, self.sigmoid_derivative(self.activelayer1))
         w1_derivative = np.dot(b1_derivative, self.input)

         #update weight and bais
         self.w1= self.w1 - 0.1 * w1_derivative
         self.w2= self.w2 - 0.1 * w2_derivative
         self.b1= self.b1 - 0.1 * b1_derivative
         self.b2= self.b2 - 0.1 * b2_derivative
        
    def train(self,inputVal,target,iteration):
        #find cost and error rate
        self.input = inputVal
        for i  in range(iteration): 
            self.feedforward()
            print ("error rate ")
            print (np.average(np.power(np.subtract(self.output , target),2)))
            self.backpropagation(target)

    def getweight_bais(self):
        return [self.w1,self.b1,self.w2,self.b2]

nn = NN()
target= np.array([[0],[1]])
#print (target.shape)
inputVal = np.array([[0],[0],[0],[1]])
#print (inputVal)

#print (nn.sigmoid(inputVal))
#print (inputVal.shape)
nn.train(inputVal,target,100)
print (nn.predict(inputVal))
        
