import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero, shape(D, C)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    cnt = 0 # number of classes didn't meet the desired margin
    for j in xrange(num_classes):
      if j == y[i]:
        continue 
        """
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        continues with the next iteration of the loop
        This is faster than put j != y[i] in the condition below
        Guess: compute margin req np indexing, which is slower than if comparison
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        cnt += 1
        loss += margin
        dW[:, j] += X[i]
    dW[:, y[i]] += - cnt * X[i]

  # Right now the loss and gradient is a sum over all training examples, 
  # but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # Add regularization to the gradient.
  dW += 2 * reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)
  margins = np.maximum(0, scores - correct_class_scores + 1)
  margins[np.arange(num_train), y] = 0 # adjustment of case j == y[i]
  loss = np.sum(margins) / num_train + reg * np.sum(W ** 2)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  """
  original solution 
  (updated since array M can be replace by indexing np.arange(num_train), y)

  M: A numpy array of shape(N, C) s.t. M[i, j] = 1 if y[i] == j; 0 otherwise
     M is the indicator of class membership of each x[i]
  N: A numpy array of shape(N, C) s.t. N[i, j] = 1 if y[i] != j; 0 otherwise
  Z: A numpy array of shape(N, C) s.t. Z[i, j] = True if margins[i, j] > 0
  Count: A numpy array of shape (N, 1),
  s.t Counter[i] is the number of classes that x[i] didn't meet the desired margin
  
  M = np.zeros((num_train, num_classes))
  M[np.arange(num_train), y] = 1
  # N = np.ones(M.shape) - M
  Z = margins > np.zeros(margins.shape) # Z * M == 0
  Count = np.sum(Z, axis = 1).reshape(num_train, 1)  
  dW = - X.T.dot(Count * M) + X.T.dot(Z) 
  # dW = - X.T.dot(Count * M) + X.T.dot(Z * N); Z * N == Z - Z * M (Z * M == 0)
  dW = dW / num_train + 2 * reg * W
  """
  # Z: A numpy array of shape(N, C) s.t. Z[i, j] = 1 if margins[i, j] > 0
  # i.e. x[i] didnt meet the desired margin for class j
  Z = np.zeros(margins.shape)
  Z[margins > 0] = 1
  Z[np.arange(num_train), y] = - np.sum(Z, axis = 1)
  dW = X.T.dot(Z) / num_train + 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
