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

caption_size = 100
image_size = 4096
embed_size = 1024
epoch = 10000


class LagrangeLoss(nn.Module):
    def __init__(self):
        super(LagrangeLoss, self).__init__()
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, distances, truth):
        distances_sm = self.softmax(distances)
        err1 = self.criterion(distances, truth)
        err2 = torch.sum(torch.abs(1 - torch.sum(distances_sm, 1)))
        err2 *= 10000
        return err1 + err2


class Hungarian(nn.Module):
    def __init__(self):
        super(Hungarian, self).__init__()

        self.fc_row = nn.Linear(n, n)
        self.fc_col = nn.Linear(n, n)
        self.fc = nn.Linear(2 * n, n)

    def forward(self, input):
        cat = torch.cat((self.fc_col(input), self.fc_row(input.transpose(1, 0))), 1)
        out = self.fc(cat)

        return out


model = Hungarian()

if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # lr=0.0002
criterion = LagrangeLoss()


def eval():
    model.eval()
    distances = autograd.Variable(torch.FloatTensor(np.random.random((n, n))))

    if torch.cuda.is_available():
        distances = distances.cuda()
    distances_np = distances.data.cpu().numpy()
    rows, hung_solution = linear_sum_assignment(distances_np)
    optimal_cost = distances_np[rows, hung_solution].sum()

    feature = model(distances)

    target = torch.autograd.Variable(torch.LongTensor(hung_solution))
    if torch.cuda.is_available():
        target = target.cuda()

    loss = criterion(feature, target)
    corrects = (torch.max(feature, 1)[1].view(target.size()).data == target.data).sum()

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

    approximate_assignment = greedy_assignment(feature, softmax=True)
    approximate_cost = 0
    for row, index in enumerate(approximate_assignment):
        approximate_cost += distances[row][index].data.cpu().numpy()[0]

    print("\n")
    print(">>>>>>>> EVALUATION {:.4f}  - {:.4f} %".format(loss.data[0], (corrects / n) * 100))
    print("Greedy cost {:.4f}  - approximate cost {:.4f} - optimal cost {:.4f}".format(
        greedy_cost, approximate_cost, optimal_cost))


def greedy_assignment(cost_matrix, softmax=False):
    num_collisions = 0
    if softmax:
        cost_matrix = torch.nn.functional.softmax(cost_matrix, dim=1)
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
    print("NUM COLLISIONS: ", num_collisions)
    return assignments


def check_bijection(assignment, solution, softmax=False):
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

    loss = criterion(feature, target)
    loss.backward()
    optimizer.step()

    corrects = (torch.max(feature, 1)[1].view(target.size()).data == target.data).sum()
    print("Epoch {:.4f} - loss {:.4f} - {:.4f} %".format(i,
                                                         loss.data[0], (corrects / n) * 100), end="\r")

    if i % 50 == 0:
        eval()
