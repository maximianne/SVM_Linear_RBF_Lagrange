# Name: Maxie Castaneda
# COMP 347 - Machine Learning
# HW No. 3

# *********************
# PROBLEMS ARE POSTED BELOW THE FUNCTIONS SECTION!!!
# *********************

# Libraries
# ------------------------------------------------------------------------------
from collections import Counter
from itertools import combinations
import numpy as np
import scipy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

# Functions
# ------------------------------------------------------------------------------


def fact(n):
    """Factorial of an integer n>=0."""
    if n in [0, 1]:
        return 1
    else:
        return n * fact(n - 1)


def partition(number: int, max_vals: tuple):
    S = set(combinations((k for i, val in enumerate(max_vals) for k in [i] * val), number))
    for s in S:
        c = Counter(s)
        yield tuple([c[n] for n in range(len(max_vals))])


def RBF_Approx(data_x, gamma, deg):
    """Transforms data in X to its RBF representation, but as an approximation
    in deg degrees.  gamma = 1/2."""
    new_X = [];
    N = data_x.shape[0];
    n = data_x.shape[1];
    count = 0
    for i in range(N):
        vec = []
        for k in range(deg + 1):
            if k == 0:
                vec += [1]
            else:
                tup = (k,) * n
                parts = list(partition(k, tup))
                for part in parts:
                    vec += [np.prod(
                        [np.sqrt(gamma ** deg) * (data_x[i, s] ** part[s]) / np.sqrt(fact(part[s])) for s in range(n)])]
        new_X += [np.exp(-gamma * LA.norm(data_x[i, :]) ** 2) * np.asarray(vec)]
        # print(str(count) + " of " + str(N))
        count += 1

    return np.asarray(new_X)


def smo_algorithm(data_x, y, C, max_iter, thresh):
    """Optimizes Lagrange multipliers in the dual formulation of SVM.
        X: The data set of size Nxn where N is the number of observations and
           n is the length of each feature vector.
        y: The class labels with values +/-1 corresponding to the feature vectors.
        C: A threshold positive value for the size of each lagrange multiplier.  
           In other words 0<= a_i <= C for each i.
        max_iter: The maximum number of successive iterations to attempt when
                  updating the multipliers.  The multipliers are randomly selected
                  as pairs a_i and a_j at each iteration and updates these according
                  to a systematic procedure of thresholding and various checks.
                  A counter is incremented if an update is less than the value
                  thresh from its previous iteration.  max_iter is the maximum
                  value this counter attains before the algorithm terminates.
        thresh: The minimum threshold difference between an update to a multiplier
                and its previous iteration.
    """
    alph = np.zeros(len(y));
    b = 0
    count = 0
    while count < max_iter:

        num_changes = 0

        for i in range(len(y)):
            w = np.dot(alph * y, data_x)
            E_i = np.dot(w, data_x[i, :]) + b - y[i]

            if (y[i] * E_i < -thresh and alph[i] < C) or (y[i] * E_i > thresh and alph[i] > 0):
                j = np.random.choice([m for m in range(len(y)) if m != i])
                E_j = np.dot(w, data_x[j, :]) + b - y[j]

                a_1old = alph[i];
                a_2old = alph[j]
                y_1 = y[i];
                y_2 = y[j]

                # Compute L and H
                if y_1 != y_2:
                    L = np.max([0, a_2old - a_1old])
                    H = np.min([C, C + a_2old - a_1old])
                elif y_1 == y_2:
                    L = np.max([0, a_1old + a_2old - C])
                    H = np.min([C, a_1old + a_2old])

                if L == H:
                    continue
                eta = 2 * np.dot(data_x[i, :], data_x[j, :]) - LA.norm(data_x[i, :]) ** 2 - LA.norm(data_x[j, :]) ** 2
                if eta >= 0:
                    continue
                # Clip value of a_2
                a_2new = a_2old - y_2 * (E_i - E_j) / eta
                if a_2new >= H:
                    a_2new = H
                elif a_2new < L:
                    a_2new = L

                if abs(a_2new - a_2old) < thresh:
                    continue

                a_1new = a_1old + y_1 * y_2 * (a_2old - a_2new)

                # Compute b
                b_1 = b - E_i - y_1 * (a_1new - a_1old) * LA.norm(data_x[i, :]) - y_2 * (a_2new - a_2old) * np.dot(
                    data_x[i, :],
                    data_x[j, :])
                b_2 = b - E_j - y_1 * (a_1new - a_1old) * np.dot(data_x[i, :], data_x[j, :]) - y_2 * (
                        a_2new - a_2old) * LA.norm(
                    data_x[j, :])

                if 0 < a_1new < C:
                    b = b_1
                elif 0 < a_2new < C:
                    b = b_2
                else:
                    b = (b_1 + b_2) / 2

                num_changes += 1
                alph[i] = a_1new
                alph[j] = a_2new

        if num_changes == 0:
            count += 1
        else:
            count = 0
        print(count)
    return alph, b


def hinge_loss_one_data(v, y, w, b):
    """ v is a single point
        y is the vector of class labels with values either +1 or -1.
        w is the support vector
        b the corresponding bias."""
    y_i = y * (np.matmul(w, v) + b)
    hinge = np.max([0.0, 1 - y_i * y])
    return hinge + (0.5 * np.linalg.norm(w) ** 2)


def hinge_loss(data_x, y, w, b):
    """Here X is assumed to be Nxn where each row is a data point of length n and N is the number of data points.
    y is the vector of class labels with values either +1 or -1.
    w is the support vector
    b the corresponding bias."""

    total_hinge = 0
    num = len(data_x)
    for i in range(num):
        total_hinge = total_hinge + hinge_loss_one_data(data_x[i], y[i], w, b)
    hinge = total_hinge
    return hinge / num


def hinge_deriv(data_x, y, w, b):
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.
    y is the vector of class labels with value either +1 or -1.
    w is the support vector
    b the corresponding bias."""
    num = len(data_x)
    deriv_w = [0, 0]
    deriv_b = 0
    for i in range(num):
        wT = w.reshape(-1, 1)
        data_x_i = data_x[i].reshape(1, -1)
        val = np.linalg.norm(np.matmul(wT, data_x_i)) + b
        if val * y[i] <= 1:
            temp_x = data_x[i]
            val2 = -y[i] * temp_x[0]
            val3 = -y[i] * temp_x[1]
            deriv_w[0] += val2 / num
            deriv_w[1] += val3 / num
            deriv_b += (-1 * y[i]) / num
    return deriv_w, deriv_b


def SVM_HL(x, y, w, e, b, iterations):
    derivW, derivB = hinge_deriv(x, y, w, b)
    for i in range(1, iterations):
        w = np.subtract(w, np.dot(e, derivW))
        b = b - e * derivB
        derivW, derivB = hinge_deriv(x, y, w, b)
    return w, b


def training_success(X_mat, Y_mat, w, e, b, iterations):
    training = []
    success = []
    numOfGroups = len(X_mat)  # 10
    for i in range(0, numOfGroups):
        add = []
        x = X_mat[i]
        y = Y_mat[i]
        w, b = SVM_HL(x, y, w, e, b, iterations)
        add.append(w)
        add.append(b)
        training.append(add)
        s = test_classification(y, w, x, b)
        success.append(s)

    return training, success


def training_successSMO(X_mat, Y_mat):
    training = []
    success = []
    numOfGroups = len(X_mat)  # 10
    for i in range(0, numOfGroups):
        add = []
        x = X_mat[i]
        y = Y_mat[i]
        # test on my computer with iterations = 5 because it could not handle 500
        alpha, b = smo_algorithm(x, y, C=0.25, max_iter=500, thresh=1e-300)
        w = get_w(alpha, y, x)
        add.append(w)
        add.append(b)
        training.append(add)
        s = test_classification(y, w, x, b)
        success.append(s)

    return training, success


def test_classification(y, w, x, b):
    num = len(x)
    c = 0
    for i in range(0, num):
        wT = w.reshape(-1, 1)
        x_i = x[i].reshape(1, -1)
        wx = np.linalg.norm(np.dot(wT, x_i))
        wx_b = wx + b
        check = y[i] * wx_b
        if check >= 1:
            c += 1
        else:
            c += 0
    return c / num


def get_w(alphas, y, x):
    m = len(x)
    w = np.zeros(x.shape[1])
    for i in range(m):
        w = w + alphas[i] * y[i] * x[i, :]
    return w


def create_3rdD(x, y):
    zList = []
    for i in range(len(x)):
        zList.append(np.sqrt(x[i] ** 2 + y[i] ** 2))
    z = np.array(zList)
    return z


# Problem #1 - Hinge Loss Optimization for SVM on Randomized Test Data
# ------------------------------------------------------------------------------
# In this problem you will be performing SVM using the hinge loss formalism as
# presented in lecture.

if __name__ == "__main__":
    # 1a. Complete the function hinge_loss and hinge_deriv in the previous section.
    # 1b. Perform SVM Using the hinge loss formalism on the data in svm_test_2.csv.
    #     Use an appropriate initialization to your gradient descent algorithm.
    data = pd.read_csv('svm_test_2.csv')
    w_0 = [8, 20]
    w_0 = np.array(w_0)
    eps = 0.1

    X = data.drop('2', axis=1)
    X = np.array(X)

    Y = data['2']
    Y = np.array(Y)

    x_plot = data['0']
    y_plot = data['1']

    w_new, b_new = SVM_HL(X, Y, w_0, eps, b=7, iterations=100)
    print("Hinge Loss SVM Classification W: ", w_new)

    b2 = w_new[1]
    m2 = w_new[0]
    equation_HL = b2 + (m2 * x_plot)

    # 1c. Perform SVM on the data in svm_test_2.csv now using the Lagrange multiplier
    #     formalism by calling the function smo_algorithm presented above.  Optimize
    #     this for values of C = 0.25, 0.5, 0.75, and 1.  I recommend taking
    #     max_iter = 2500 and thresh = 1e-5 when calling the smo_algorithm.

    alpha, b_l = smo_algorithm(X, Y, C=0.25, max_iter=2500, thresh=0.0001)
    w_l1 = get_w(alpha, Y, X)
    print(w_l1)
    bl_1 = w_l1[0]+b_l
    ml_1 = w_l1[1]
    equation_L1 = bl_1 + (ml_1 * x_plot)

    alpha2, b_l2 = smo_algorithm(X, Y, C=0.5, max_iter=2500, thresh=0.000001)
    w_l2 = get_w(alpha2, Y, X)
    print(w_l2)
    bl_2 = w_l2[1]+b_l2
    ml_2 = w_l2[0]
    equation_L2 = bl_2 + (ml_2 * x_plot)

    alpha3, b_l3 = smo_algorithm(X, Y, C=0.75, max_iter=2500, thresh=0.000001)
    w_l3 = get_w(alpha3, Y, X)
    print(w_l3)
    bl_3 = w_l3[1]+b_l3
    ml_3 = w_l3[0]
    equation_L3 = bl_3 + (ml_3 * x_plot)

    alpha4, b_l4 = smo_algorithm(X, Y, C=1, max_iter=2500, thresh=0.000001)
    w_l4 = get_w(alpha4, Y, X)
    print(w_l4)
    bl_4 = w_l4[1]+b_l4
    ml_4 = w_l4[0]
    equation_L4 = bl_4 + (ml_4 * x_plot)

    # 1d. Make a scatter plot of the data with decision boundary lines indicating
    #     the hinge model, and the various Lagrange models found from part c.  Make
    #     sure you have a legend displaying each one clearly and adjust the transparency
    #     as needed.
    plt.title('Decision Boundary Lines - Hinge and Lagrange')
    myCol = colors.ListedColormap(['purple', 'black'])
    plt.scatter(x_plot, y_plot, c=y_plot, cmap=myCol)
    plt.plot(x_plot, equation_HL, label='Hinge-Classification')
    plt.plot(x_plot, equation_L1, label='Lagrange, C = 0.25')
    plt.plot(x_plot, equation_L2, label='Lagrange, C = 0.5')
    plt.plot(x_plot, equation_L3, label='Lagrange, C = 0.75')
    plt.plot(x_plot, equation_L4, label='Lagrange, C = 1')
    plt.legend()
    plt.show()

    # 1e. Perform SVM on the radial data, but preprocess the data by using a kernel
    #     embedding of the data into 3 dimensions.  This can be accomplished by
    #     taking z = sqrt(x**2 + y**2) for each data point.  Learn an optimal model
    #     for separating the data using the Lagrange multiplier formalism.  Experiment
    #     with choices for C, max_iter, and thresh as desired.'''

    data2 = pd.read_csv('radial_data.csv')
    x_plot = data2['0']
    x_plot = np.array(x_plot)
    y_plot = data2['1']
    y_plot = np.array(y_plot)
    z = create_3rdD(x_plot, y_plot)
    data2['3'] = z.tolist()

    X = data2.drop('2', axis=1)
    X = np.array(X)

    Y = data2['2']
    Y = np.array(Y)

    alpha, beta = smo_algorithm(X, Y, C=1.5, max_iter=200, thresh=0.01)
    w = get_w(alpha, Y, X)

    print(w)
    print(test_classification(Y, w, X, beta))

    w2 = w[2]  # z
    w1 = w[1]  # y
    w0 = w[0]  # x

    x_surf, y_surf = np.meshgrid(x_plot, y_plot)
    Z = w[0] * x_surf + w[1] * y_surf + w[2]

    ax = plt.axes(projection='3d')
    ax.view_init(0, 55)
    myCol = colors.ListedColormap(['purple', 'black'])
    ax.scatter3D(x_plot, y_plot, z, c=z, cmap=myCol)
    ax.plot_surface(x_surf, y_surf, Z)
    plt.show()

    # Problem #2 - Cross Validation and Testing for Breast Cancer Data
    # ------------------------------------------------------------------------------
    # In this problem you will use the breast cancer data in an attempt to use SVM
    # for a real-world classification problem.

    # 2a. Pre-process the data so that you separate the main variables.  Your data
    #     X should consist all but the first two and very last columns in the dataframe.
    #     Create a variable Y to reinterpret the binary classifiers 'B' and 'M' as
    #     -1 and +1, respectively.

    data3 = pd.read_csv('breast_cancer.csv')
    # SHUFFLE DATAFRAME CONTENT
    data3 = data3.sample(frac=1)
    data3['diagnosis'].replace({"M": -1}, inplace=True)
    data3['diagnosis'].replace({"B": 1}, inplace=True)

    X = data3.drop(columns=['diagnosis', 'id', 'Unnamed: 32'], axis=1)
    X = np.array(X)
    # print(X)

    Y = data3['diagnosis']
    Y = np.array(Y)

    # 2b. Perform cross-validation using a linear SVM model on the data by dividing
    #     the indexes into 10 separate randomized classes (I recommend looking up
    #     np.random.shuffle and np.array_split).  Make sure you do the following:

    cases_X = np.array_split(X, 10)
    cases_Y = np.array_split(Y, 10)

    #       1. Make two empty lists, Trained_models and Success_rates.  In Trained_models
    #          save ordered pairs of the learned models [w,b] for each set of
    #          training data.  In Success_rates, save the percentage of successfully
    #          classified test points from the remaining partition of the data.
    #          Remember that the test for correct classification is that y(<w,x> + b) >= 1.

    #       2. Make a histogram of your success rates.  Don't expect these to be stellar
    #          numbers.  They will most likely be abysmal success rates.  Unfortunately
    #          SVM is a difficult task to optimize by hand, which is why we are fortunate
    #          to have kernel methods at our disposal.  Speaking of which.....

    w = [10, -9]
    w = np.array(w)
    e = 0.1
    b = 2
    iterations = 300

    '''Trained_models, Success_rates = training_success(cases_X, cases_Y, w, e, b, iterations)
    plt.title('Success Rates for Breast Cancer Data - Linear via SVM Hingeloss ')
    plt.locator_params(axis="y", integer=True, tight=True)
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')
    plt.hist(Success_rates, bins=6)
    plt.show()'''

    Trained_modelsSMO, Success_rates = training_successSMO(cases_X, cases_Y)
    plt.title('Success Rates for Breast Cancer Data - Linear via SVM Lagrange multiplier ')
    plt.locator_params(axis="y", integer=True, tight=True)
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')
    plt.hist(Success_rates, bins=6)
    plt.show()

    # 2c. Repeat cross-validation on the breast cancer data, but instead of a linear
    #     SVM model, employ an approximation to the RBF kernel as discussed in class.
    #     Note that what this does is that it transforms the original data x into a
    #     variable X where the data is embedded in a higher dimension.  Generally when
    #     data gets embedded in higher dimensions, there's more room for it to be spaced
    #     out in and therefore increases the chances that your data will be linearly
    #     separable.  Do this for deg = 2,3.  I recommend taking gamma = 1e-6.
    #     Don't be surprised if this all takes well over an hour to terminate.

    # -------- RBF DEG = 2 ----------- #
    X_newX = RBF_Approx(X, gamma=1e-6, deg=2)
    cases_X = np.array_split(X_newX, 10)
    cases_Y = np.array_split(Y, 10)

    '''Trained_modelRBF2, Success_rates = training_success(cases_X, cases_Y, w, e, b, iterations)
    plt.title('Success Rates for Breast Cancer Data (HL) - RBF Kernel, Deg = 2')
    plt.locator_params(axis="y", integer=True, tight=True)
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')
    plt.hist(Success_rates, bins=6)
    plt.show()'''

    Trained_modelsRBF2SMO, Success_rates = training_successSMO(cases_X, cases_Y)
    plt.title('Success Rates for Breast Cancer Data (SMO) - RBF Kernel, Deg = 2')
    plt.locator_params(axis="y", integer=True, tight=True)
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')
    plt.hist(Success_rates, bins=6)
    plt.show()

    # ------ RBF DEG 3 --------- #
    X_newX = RBF_Approx(X, gamma=1e-6, deg=3)
    cases_X = np.array_split(X_newX, 10)
    cases_Y = np.array_split(Y, 10)

    '''Trained_modelsRBF3, Success_rates2 = training_success(cases_X, cases_Y, w, e, b, iterations)
    plt.title('Success Rates for Breast Cancer Data (HL) - RBF Kernel, Deg = 3')
    plt.locator_params(axis="y", integer=True, tight=True)
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')
    plt.hist(Success_rates2, bins=6)
    plt.show()'''

    Trained_modelsRBF32SMO, Success_rates2 = training_successSMO(cases_X, cases_Y)
    plt.title('Success Rates for Breast Cancer Data (SMO) - RBF Kernel, Deg = 3')
    plt.locator_params(axis="y", integer=True, tight=True)
    plt.xlabel('Success Rate')
    plt.ylabel('Frequency')
    plt.hist(Success_rates2, bins=6)
    plt.show()

# Notes for Problem #2:
# 1. To save yourself from writing the same code twice, I recommend making this
#    type of if/else statement before performing SVM on the breast cancer data:

#        METHOD = ''
#        if METHOD == 'Lin':
#            X = x
#        elif METHOD == 'RBF':
#            deg = 2; gamma = 1e-6
#            X = RBF_Approx(x,gamma,deg)

# 2. For implementing smo_algorithm for the breast cancer data, I recommend
#    taking max_iter = 500 and thresh = 1e-300
