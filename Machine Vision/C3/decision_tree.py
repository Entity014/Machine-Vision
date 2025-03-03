import numpy as np
import matplotlib.pyplot as plt

def decision_line_x1(data, value):
    return len(data[data[:, 0] < value]), len(data[data[:, 0] > value])

def decision_line_x2(data, value):
    return len(data[data[:, 1] < value]), len(data[data[:, 1] > value])

def decision_tree_x1():
    for i in range(1, 6):
        sum_child1, sum_child2 = [0, 0]
        for value in parent:
            sum_child1 += decision_line_x1(value, i)[0]
            sum_child2 += decision_line_x1(value, i)[1]

        entropy_child1_cal, entropy_child2_cal = [0, 0]
        for value in parent:
            if decision_line_x1(value, i)[0] == 0 or decision_line_x1(value, i)[1] == 0:
                continue
            entropy_child1_cal += (decision_line_x1(value, i)[0] / sum_child1) * np.log2(decision_line_x1(value, i)[0] / sum_child1)
            entropy_child2_cal += (decision_line_x1(value, i)[1] / sum_child2) * np.log2(decision_line_x1(value, i)[1] / sum_child2)
        entropy_child1 = -entropy_child1_cal
        entropy_child2 = -entropy_child2_cal

        ig = entropy_parent - (((sum_child1 / (sum_child1 + sum_child2)) * entropy_child1) + ((sum_child2 / (sum_child1 + sum_child2)) * entropy_child2))
        print(f"Entropy x1 > {i} : {ig}")

def decision_tree_x2():
    for i in range(1, 6):
        sum_child1, sum_child2 = [0, 0]
        for value in parent:
            sum_child1 += decision_line_x2(value, i)[0]
            sum_child2 += decision_line_x2(value, i)[1]

        entropy_child1_cal, entropy_child2_cal = [0, 0]
        for value in parent:
            if (decision_line_x2(value, i)[0] / sum_child1 == 0) or (decision_line_x2(value, i)[1] / sum_child2 == 0):
                continue
            entropy_child1_cal += (decision_line_x2(value, i)[0] / sum_child1) * np.log2(decision_line_x2(value, i)[0] / sum_child1)
            entropy_child2_cal += (decision_line_x2(value, i)[1] / sum_child2) * np.log2(decision_line_x2(value, i)[1] / sum_child2)
        entropy_child1 = -entropy_child1_cal
        entropy_child2 = -entropy_child2_cal

        ig = entropy_parent - (((sum_child1 / (sum_child1 + sum_child2)) * entropy_child1) + ((sum_child2 / (sum_child1 + sum_child2)) * entropy_child2))
        print(f"Entropy x2 > {i} : {ig}")

# Define the data
class_1 = np.array([(0.5, 3.5), (0.5, 5.5), (1.5, 3.25), (1.5, 3.75), (1.5, 4.5), (3.5, 2.5), (4.5, 2.25)])
class_2 = np.array([(0.5, 1.5), (1.5, 2.5), (2.5, 1.5), (2.5, 3.5), (4.5, 1.25), (4.5, 1.75), (5.5, 0.5)])
class_3 = np.array([(3.5, 4.5), (4.5, 2.75), (4.5, 4.25), (4.5, 4.75), (4.5, 5.5), (5.5, 4.5), (5.5, 5.5)])

parent = np.array([class_1, class_2, class_3])

entropy = 0
for value in parent:
    entropy += (len(value) / (parent.shape[0] * parent.shape[1])) * np.log2(len(value) / (parent.shape[0] * parent.shape[1]))
entropy_parent = -entropy

decision_tree_x1()
decision_tree_x2()
plt.scatter(class_1[:, 0], class_1[:, 1], color='green', label='Class 1')
plt.scatter(class_2[:, 0], class_2[:, 1], color='red', label='Class 2')
plt.scatter(class_3[:, 0], class_3[:, 1], color='purple', label='Class 3')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of x1 vs x2')
plt.legend()
plt.grid(True)


plt.xlim(0, 6)
plt.ylim(0, 6)

# Show the plot
plt.show()
