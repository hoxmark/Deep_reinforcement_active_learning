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


class LagrangeLoss(nn.Module):
    def __init__(self):
        super(LagrangeLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, distances, truth):
        err1 = self.criterion(distances, truth)
        err2 = torch.sum(torch.abs(1 - torch.sum(distances, 1)))
        err3 = torch.sum(torch.abs(1 - torch.sum(distances, 0)))
        # print(err2)
        return err1 + err2

class Hungarian(nn.Module):
    def __init__(self):
        super(Hungarian, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(n*n, n*n)

    def forward(self, input):
        out = input.view(1, n * n)
        # out = self.softmax(out)
        out = self.fc(out)
        out = out.view(n, n)
        return out

def eval():
    model.eval()
    distances = autograd.Variable(torch.FloatTensor(np.random.random((n, n))))

    if torch.cuda.is_available():
        distances = distances.cuda()
    distances_np = distances.data.cpu().numpy()

    # Solve using the hungarian algorithm, and get the associated cost
    rows, hung_solution = linear_sum_assignment(distances_np)
    optimal_cost = distances_np[rows, hung_solution].sum()

    # Get our
    feature = model(distances)
    target = torch.autograd.Variable(torch.LongTensor(hung_solution))
    if torch.cuda.is_available():
        target = target.cuda()

    loss = criterion(feature, target)

    corrects = (torch.max(feature, 1)[1].view(target.size()).data == target.data).sum()

    # Get the cost of our predicted matrix
    distances_np_pred = feature.data.cpu().numpy()
    rows_pred, hung_solution_pred = linear_sum_assignment(distances_np_pred)
    cost_pred = distances_np[rows_pred, hung_solution_pred].sum()

    print("\n")
    # print(feature)
    check_bijection(feature, hung_solution, softmax=True)

    greedy_ass = greedy_assignment(distances)

    greedy_cost = 0
    for row, index in enumerate(greedy_ass):
        greedy_cost += distances[row][index].data.cpu().numpy()[0]

    approximate_assignment = greedy_assignment(feature)
    approximate_cost = 0
    for row, index in enumerate(approximate_assignment):
        approximate_cost += distances[row][index].data.cpu().numpy()[0]

    print("\n")
    print(">>>>>>>> EVALUATION {:.4f}  - {:.4f} %".format(loss.data[0], (corrects / n) * 100))
    print("Greedy cost {:.4f}  - approximate cost {:.4f} - optimal cost {:.4f}".format(
        greedy_cost, approximate_cost, optimal_cost))


def greedy_assignment(cost_matrix, print_col=True):
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
    if print_col:
        print("NUM COLLISIONS: ", num_collisions)
    return assignments


def check_bijection(assignment, solution, softmax=False):
    softmax = False
    errors = []
    if softmax:
        assignment = torch.nn.functional.softmax(assignment)
    assignments = torch.max(assignment, 1)[1].data.cpu().numpy()
    for index, assignment in enumerate(assignments):
        error = False
        for assignment2 in np.delete(assignments, index):
            if assignment == assignment2:
                error = True
                break
        errors.append(error)

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

model = Hungarian()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # lr=0.0002
criterion = LagrangeLoss()

model.train()
for i in range(epoch):
    distances = autograd.Variable(torch.FloatTensor(np.random.random((n, n))))
    if torch.cuda.is_available():
        distances = distances.cuda()
    distances_np = distances.data.cpu().numpy()
    rows, hung_solution = linear_sum_assignment(distances_np)

    optimizer.zero_grad()

    feature = model(distances)
    target = torch.autograd.Variable(torch.LongTensor(hung_solution))

    if torch.cuda.is_available():
        target = target.cuda()

    # print(feature)

    loss = criterion(feature, target)
    loss.backward()
    optimizer.step()

    # corrects = (torch.max(feature, 1)[1].view(target.size()).data == target.data).sum()
    approximate_assignment = greedy_assignment(feature, print_col=False)
    corrects = sum([1 if approximate_assignment[i] == hung_solution[i] else 0 for i, _ in enumerate(approximate_assignment)])
    print("Epoch {:.4f} - loss {:.4f} - {:.4f} %".format(i,
                                                         loss.data[0], (corrects / n) * 100), end="\r")

    if i % 50 == 0:
        eval()
