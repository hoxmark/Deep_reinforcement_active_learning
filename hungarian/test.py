import torch
from random import shuffle
from torch import autograd
from torch import nn
import numpy as np

from scipy.optimize import linear_sum_assignment

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


n = 100
epoch = 10000


class CostLoss(nn.Module):
    def __init__(self):
        super(CostLoss, self).__init__()

    def forward(self, feature, distances, target):
        approximate_assignment = torch.max(feature, 1)[1]
        cost_vector = distances.gather(1, approximate_assignment.unsqueeze(1)).squeeze()
        approximate_cost = torch.sum(cost_vector)
        err2 = torch.sum(torch.abs(1 - torch.sum(feature, 1)))

        return approximate_cost + err2


class LagrangeLoss(nn.Module):
    def __init__(self):
        super(LagrangeLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, distances, target):
        err1 = self.criterion(feature, target)
        err2 = torch.sum(torch.abs(1 - torch.sum(feature, 1)))
        err3 = torch.sum(torch.abs(1 - torch.sum(feature, 0)))
        return err1 + err2 + err3

class Hungarian(nn.Module):
    def __init__(self):
        super(Hungarian, self).__init__()
        self.softmax = nn.Softmax()
        self.fc = nn.Linear(n*n, n*n)
        self.relu = nn.ReLU()


    def forward(self, input):
        out = input.view(1, n * n)
        out = self.fc(out)
        # out = self.relu(out)
        # out = self.softmax(out)
        out = out.view(n, n)
        return out

def eval():
    model.eval()
    # distances = autograd.Variable(torch.FloatTensor(np.random.random((n, n))))
    #
    # if torch.cuda.is_available():
    #     distances = distances.cuda()
    # distances_np = distances.data.cpu().numpy()
    #
    # # Solve using the hungarian algorithm, and get the associated cost
    # rows, hung_solution = linear_sum_assignment(distances_np)
    optimal_cost = distances_np[rows, hung_solution].sum()

    # Get our
    feature = model(distances)
    target = torch.autograd.Variable(torch.LongTensor(hung_solution))
    if torch.cuda.is_available():
        target = target.cuda()

    # loss = criterion(feature, target)
    loss = criterion(feature, distances, target)
    corrects = (torch.max(feature, 1)[1].view(target.size()).data == target.data).sum()

    # Get the cost of our predicted matrix
    distances_np_pred = feature.data.cpu().numpy()
    rows_pred, hung_solution_pred = linear_sum_assignment(distances_np_pred)
    cost_pred = distances_np[rows_pred, hung_solution_pred].sum()

    print("\n")
    check_bijection(feature, hung_solution)

    greedy_ass = greedy_assignment(distances)
    greedy_cost = 0
    for row, index in enumerate(greedy_ass):
        greedy_cost += distances[row][index].data.cpu().numpy()[0]

    approximate_assignment = torch.max(feature, 1)[1].data.cpu().numpy()
    approximate_cost = 0
    for row, index in enumerate(approximate_assignment):
        approximate_cost += distances[row][index].data.cpu().numpy()[0]

    print("\n")
    print(">>>>>>>> EVALUATION {:.4f}  - {:.4f} %".format(loss.data[0], (corrects / n) * 100))
    print("Greedy cost {:.4f}  - approximate cost {:.4f} - optimal cost {:.4f}".format(
        greedy_cost, approximate_cost, optimal_cost))


def greedy_assignment(cost_matrix):
    # print(cost_matrix)
    num_collisions = 0
    assignments = []
    cost_matrix = cost_matrix.data.cpu().numpy()
    for row in cost_matrix:
        best_assignments = np.argsort(row)
        assignment = best_assignments[0]

        while assignment in assignments:
            num_collisions += 1
            best_assignments = np.delete(best_assignments, 0)
            assignment = best_assignments[0]

        assignments.append(assignment)
    return assignments


def check_bijection(assignment, solution):
    errors = []
    assignments = torch.max(assignment, 1)[1].data.cpu().numpy()
    for index, assignment in enumerate(assignments):
        error = False
        for assignment2 in np.delete(assignments, index):
            if assignment == assignment2:
                error = True
                break
        errors.append(error)

    print("\t \t Unique? \t\t Correct? ")
    print("="*60)
    for index, assignment in enumerate(assignments):
        error = errors[index]
        ass = "Image {} \t Caption {}".format(index, assignment)
        if error:
            ass = bcolors.FAIL + ass + bcolors.ENDC
        else:
            ass = bcolors.OKGREEN + ass + bcolors.ENDC

        correct = "Caption {}".format(solution[index])

        if assignment == solution[index]:
            correct = bcolors.OKGREEN + correct + bcolors.ENDC
        else:
            correct = bcolors.FAIL + correct + bcolors.ENDC

        line = ass + "\t|\t" + correct
        print(line)
        # print("-"*60)

model = Hungarian()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters())  # lr=0.0002
criterion = LagrangeLoss()
# criterion = CostLoss()
model.train()

# Create one training example to use
distances = autograd.Variable(torch.FloatTensor(np.random.random((n, n))), requires_grad=True)
if torch.cuda.is_available():
    distances = distances.cuda()
distances_np = distances.data.cpu().numpy()
rows, hung_solution = linear_sum_assignment(distances_np)

for i in range(epoch):
    optimizer.zero_grad()
    #
    # distances = autograd.Variable(torch.FloatTensor(np.random.random((n, n))), requires_grad=True)
    # if torch.cuda.is_available():
    #     distances = distances.cuda()
    # distances_np = distances.data.cpu().numpy()
    # rows, hung_solution = linear_sum_assignment(distances_np)
    optimal_cost = distances_np[rows, hung_solution].sum()

    feature = model(distances)
    target = torch.autograd.Variable(torch.LongTensor(hung_solution))

    if torch.cuda.is_available():
        target = target.cuda()

    loss = criterion(feature, distances, target)
    loss.backward()
    optimizer.step()

    approximate_assignment = torch.max(feature, 1)[1].data.cpu().numpy()
    corrects = sum([1 if approximate_assignment[i] == hung_solution[i] else 0 for i, _ in enumerate(approximate_assignment)])
    print("Epoch {:.4f} - loss {:.4f} - {:.4f} %".format(i,
                                                         loss.data[0], (corrects / n) * 100), end="\r")

    if i % 50 == 0:
        eval()
