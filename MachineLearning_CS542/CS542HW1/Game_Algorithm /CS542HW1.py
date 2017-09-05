import numpy as np


#formatting data to be able to use them properly

reading = open('game.txt')

previous = []
prediction = []
newPlay = []

def read2(f):
    for line in f:
        try:
            line2 = f.next()
        except StopIteration:
            line2 = ''

        yield line, line2




for line, line2 in read2(reading):
    data = line.split()
    previous.append(data)
    data = line2.split()
    prediction.append(data)


reading = open('newTurn.txt')

for line in reading:
    data = line.split()
    newPlay = data





x = np.array(previous, dtype = float)
y = np.array(prediction, dtype = float)
play = np.array(newPlay, dtype = float)



class Neural_Network(object):
	def __init__(self):
	#Define HyperParameters
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3
		#Weights (parameters)
		self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

	def forward(self, x):
	#propagate inputs through network
		self.z2 = np.dot(x, self.w1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.w2)
		yHat = self.sigmoid(self.z3)

		return yHat

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		#derivative of Sigmoid Function
		return np.exp(-z)/((1+np.exp(-z))**2)

	def costFunction(self, y , yHat):
		cost = sum(0.5*(y-yHat)**2)

		return cost

	def costFunctionPrime(self, x, y):
		#Compute derivative with respect to W1 and W2
		self.yHat = self.forward(x)
		delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.w2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(x.T, delta2)

		return dJdW1, dJdW2


pattern = Neural_Network()

print(" Rock = 0.33,  Paper = 0.66, Scissor = 1 ")
print("the initial data are ")
print(y)



for _ in range(100000):

	dJdW1, dJdW2 = pattern.costFunctionPrime(x,y)
	learningRate = 1
	pattern.w1 = pattern.w1 - learningRate*dJdW1
	pattern.w2 = pattern.w2 - learningRate*dJdW2



print("the new trained predicted result is ")

print(pattern.forward(x))

print("the possible value of the computer move in the next round is ")
print(pattern.forward(play))

print("This is the possible value of the next move of the computer, play the winning move based on that")




#print( pattern.forward(previousPlay))
