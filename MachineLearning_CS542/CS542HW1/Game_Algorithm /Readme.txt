Building a neural network

type of inputs

rock, paper , scissors

rock = 0.33
paper = 0.66
scissor = 1

2 players: me and the computer

Neural network that takes two inputs, my play and the computer play

input layer = 2 perceptrons
hidden layer = 3 perceptrons
output layer = 1 perceptrons


2 inputs:
my play and the computer play on the previous round

output is the hand the computer would play depending on the previous round


Strategy to use my neural network
To train the neural network
input all the previous data except the last play but also input the last move of the computer in the last round

To test it
input in the neural network the last round and get the output from the neural network which will be the possible move of the computer in the next round


Example on how to use it.

Lets say we have 3 rounds and we want to predict the move of the computer in the 4th round

          Payer   Computer
round 1   S        P            ==> 1     0.66
round 2   R        P            ==> 0.33  0.66
round 3   P        R            ==> 0.66  0.33


How to determine the 4th round:
in game.txt

1  0.66
0.66
0.33 0.66
0.33

in newTurn.txt

0.66  0.33


And then just run the algorithm and look for the value corresponding to the possible play of the computer
