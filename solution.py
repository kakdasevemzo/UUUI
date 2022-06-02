import numpy as np
import sys
import os
import math
import random
import time

from numpy.random import default_rng

from utils import check_file_exist, sigmoid_function

TRAINING_FILE_NAME = ""
TESTING_FILE_NAME = ""
NEURAL_NETWORK_STRUCTURE = ""
POPULATION_SIZE = 0
ELITISM = 0
MUTATION_PROBABILITY = 0
ST_GAUSS_DEVIATION = 0
NUMBER_OF_ITERATIONS = 0
args = sys.argv
data_directory = os.path.abspath('data')

print(data_directory)
for x in range(0, len(args)):
    if args[x] == "--train":
        TRAINING_FILE_NAME = args[x + 1]
    elif args[x] == "--test":
        TESTING_FILE_NAME = args[x + 1]
    elif args[x] == "--nn":
        NEURAL_NETWORK_STRUCTURE = args[x + 1]
    elif args[x] == "--popsize":
        POPULATION_SIZE = int(args[x + 1])
    elif args[x] == "--elitism":
        ELITISM = int(args[x + 1])
    elif args[x] == "--p":
        MUTATION_PROBABILITY = float(args[x + 1])
    elif args[x] == "--K":
        ST_GAUSS_DEVIATION = float(args[x + 1])
    elif args[x] == "--iter":
        NUMBER_OF_ITERATIONS = int(args[x + 1])

# getting train and test file paths
train_file_path = os.path.join(data_directory, TRAINING_FILE_NAME)
test_file_path = os.path.join(data_directory, TRAINING_FILE_NAME)
print("Train file:", check_file_exist(train_file_path))
print("Test file:", check_file_exist(test_file_path))

# reading data from train file
train_file = open(train_file_path, 'r', encoding="utf-8")
train_data = train_file.readlines()
train_file.close()
train_data.pop(0)

# reading data from test file
test_file = open(test_file_path, 'r', encoding="utf-8")
test_data = test_file.readlines()
test_file.close()

# getting input dimensions
input_dimensions = len(train_data[0].split(',')) - 1
print("Input dimensions:", input_dimensions)

# getting number of neurons in each layer and number of layers
network_layers = NEURAL_NETWORK_STRUCTURE.split('s')[:-1]
print()
for index, layer in enumerate(network_layers):
    print("Number of neurons in " + str(index + 1) + ". layer:", layer)

# adds output layer to the neural network, output layer set to 1
network_layers.append("1")

rng = default_rng()
mu, sigma = 0, 0.01
weights_list = []
weights = []

current_population = []
count = 1
# generating random weights from normal distribution for each layer
while count <= POPULATION_SIZE:
    for index, layer in enumerate(network_layers):
        weights = []
        if index == 0:
            for i in range(0, int(layer)):
                weights.append(rng.normal(mu, sigma, size=int(input_dimensions) + 1))
            weights_list.append(np.array(weights))
        else:
            for i in range(0, int(network_layers[index])):
                weights.append(rng.normal(mu, sigma, size=int(network_layers[index - 1]) + 1))
            weights_list.append(np.array(weights))
    current_population.append(weights_list)
    weights_list = []
    count = count + 1


# print(weights_list[0][:,n]) access n-th column of 1st layer_weight


def fit(train_data, weights_list):
    err = 0
    output_data = []
    for line in train_data:
        train_row = line.split(',')  # [x1,x2,y/n]
        train_row[-1] = train_row[-1][:-1]  # [x1,x2,y]
        input_array = np.array(train_row[:-1])  # [x1,x2]
        input_array = input_array.astype(np.float64) #????
        for k in range(0, len(weights_list)):
            weights0 = weights_list[k][:, -1]  # gets last column of weights list
            weightsN = weights_list[k][:, 0:-1]
            if k == 0:
                newArray = np.matmul(weightsN, input_array)
            else:
                newArray = np.matmul(weightsN, layer_output)
            newArray = np.add(newArray, weights0)
            if k != len(weights_list) - 1:
                layer_output = sigmoid_function(newArray)
        err = err + math.pow(newArray[0] - float(train_row[-1]), 2)
        output_data.append(newArray[0])
    err = err / len(output_data)
    return err


def evaluate(current_generation):
    errors = {}
    for sample in current_generation:
        data = fit(train_data, sample)
        errors[data] = sample
    dictionary_items = errors.items()
    sorted_items = sorted(dictionary_items)

    return sorted_items


def getFitnessSum(fitness_result):
    sum = 0
    for element in fitness_result:
        sum = sum + 1 / element[0]
    return sum


def cross_and_mutate(first_weights, second_weights):
    res_weights = []
    for i in range(0, len(first_weights)):
        res_weight = np.add(first_weights[i], second_weights[i])
        res_weight = res_weight / 2
        for index, element in enumerate(res_weight):
            rand = random.uniform(0, 1)
            if rand < MUTATION_PROBABILITY:
                res_weight[index] = element + rng.normal(mu, ST_GAUSS_DEVIATION)
        res_weights.append(res_weight)
    return res_weights


iteration = 0

while iteration <= NUMBER_OF_ITERATIONS:
    res = evaluate(current_population)
    if iteration % 2000 == 0 and iteration != 0:
        print(res[0][0])
    new_generation = []
    for k in range(0, ELITISM):
        new_generation.append(res[k][1])
    for k in range(0, POPULATION_SIZE - ELITISM):
        wheel = []
        fitnessSum = getFitnessSum(res)
        wheel.append((1 / res[0][0]) / fitnessSum)
        for i in range(1, POPULATION_SIZE):
            if i == POPULATION_SIZE - 1:
                wheel.append(1)
            else:
                wheel.append(wheel[i - 1] + (1 / res[i][0]) / fitnessSum)
        first_random = random.uniform(0, 1)
        second_random = random.uniform(0, 1)
        first_parent = -1
        second_parent = -1
        for i in range(0, len(wheel)):
            if first_random < wheel[i] and first_parent == -1:
                first_parent = i
            if second_random < wheel[i] and second_parent == -1:
                second_parent = i
        first_weights = res[first_parent][1]
        second_weights = res[second_parent][1]
        child = cross_and_mutate(first_weights, second_weights)
        new_generation.append(child)
    current_population = new_generation
    iteration = iteration + 1
    if iteration == 1000:
        break
