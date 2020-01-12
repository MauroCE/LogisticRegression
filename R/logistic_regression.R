run <- function(lr) UseMethod("run")
grad_ascent <- function(lr) UseMethod("grad_ascent")
newton_method <- function(lr) UseMethod("newton_method")

# TODO: Put this somewhere else. CBA at 3:30am
sigmoid <- function(x) 1.0 / (1.0 + exp(-x))

#' Logistic Regression S3 Object
#'
#' @param X Matrix of training examples of dimensions (number of obs, number of features + 1). The
#' first column must be a column of 1s to fit the intercept.
#' @param y Column vector of 0-1 training labels of dimension (number of obs, 1).
#' @param cost String indicating which cost function to optimize. Options are "MLE" or "MAP". If "MAP" is
#' chosen an isotropic gaussian centered at zero and with `sigmab^2 * diag(ncol(X))` as its variance-covariance
#' matrix is placed as a prior on the coefficients. This corresponds to Ridge regularization on **all** the
#' coefficients, including the intercept.
#' @param method String indicating the optimization method used to optimize `cost`. If method is `BFGS` then
#' the function `optim()` is used. Otherwise, class methods are implemented for `grad_ascent()` (performing
#' gradient ascent) and `newton_method()`. In case they are implemented both for the `MLE` case and the `MAP`
#' case.
#' @param sigmab Standard deviation of the univariate gaussian distribution placed on each coordinate of the
#' vector of coefficients. It's the inverse of the regularization parameter. Should not be zero.
#' @param niter Number of iterations that the optimization algorithm should perform. This is passed only to
#' `grad_ascent()` and `newton_method()`, but not to the `optim()` function.
#' @param alpha Learning rate for `newton_method()`. Used to dump or enhance learning to avoid missing or
#' not reaching the optimal solution. Could be merged with `gamma` but defaults are different.
#' @param gamma Learning rate for `grad_ascent()`, used to dump or enhance learning to avoid missing or
#' or not reaching the optimal solution.
#' @return
#' @export
#'
#' @examples
logistic_regression <- function(X, y, cost="MLE", method="BFGS", sigmab=1.0, niter=100,
                                alpha=0.1, gamma=0.001){
  start <- matrix(0, nrow=ncol(X))
  # Define cost functions
  mle_cost <- function(beta) sum(log(1 + exp((1 - 2*y) * (X %*% beta))))
  map_cost <- function(beta) (sigmab^2)*mle_cost(beta) + 0.5*sum(beta^2)
  # Determine and define selected Cost Function
  if      (cost == "MLE") costfunc <- mle_cost
  else if (cost == "MAP") costfunc <- map_cost
  # S3 object creation
  lr <- list(start=start, X=X, y=y, cost=cost, method=method, sigmab=sigmab, niter=niter,
             alpha=alpha, gamma=gamma, costfunc=costfunc, beta=NULL, hessian=NULL)
  # Run optimization
  if (method=="BFGS") {
    result <- optim(par=start, fn=costfunc, method=method, hessian=TRUE)
    lr$beta    <- result$par
    lr$hessian <- result$hessian # to be used in other portfolios
  } else if (method=="GA") {
    lr$beta <- grad_ascent(lr)
  } else if (method=="NEWTON") {
    lr$beta <- newton_method(lr)
  }
  class(lr) <- "logistic_regression"
  return(lr)
}


#' Pretty-Print method for S3 Object `"logistic_regression"`.
#'
#' @param x Instance of class \code{\link{logistic_regression}}.
#' @param ... Additional arguments (not implemented yet).
#'
#' @return None. But as a side effect prints to sd.out.
#' @export
#'
#' @examples
print.logistic_regression <- function(x, ...){
    cat("S3 Object of Class logistic_regression.\n")
    cat("Cost Function:        ", x$cost, "\n")
    cat("Optimization Method:  ", x$method, "\n")
    cat("Solution:             ", x$beta, "\n")
  }


grad_ascent<- function(lr){
  beta <- lr$start
  if (lr$cost=="MLE"){
    for (i in 1:lr$niter) {
      beta <- beta + lr$gamma * t(lr$X) %*% (lr$y - sigmoid(lr$X %*% beta))
    }
  } else if (lr$cost=="MAP"){
    for (i in 1:lr$niter) {
      beta <- beta + lr$gamma*(lr$sigmab^2*t(lr$X) %*% (lr$y - sigmoid(lr$X %*% beta)) - beta)
    }
  }
  return(beta)
}


newton_method <- function(lr){
  # Learning rate is suggested at 0.1. For 1.0 standard Newton method is recovered
  beta <- lr$start
  if (lr$cost=="MLE"){
    for (i in 1:lr$niter){
      D_k <- diag(drop(sigmoid(lr$X%*%beta)*(1 - sigmoid(lr$X%*%beta))))
      d_k <- solve(t(lr$X)%*%D_k %*% lr$X, lr$alpha*t(lr$X) %*% (lr$y - sigmoid(lr$X %*% beta)))
      beta <- beta + d_k
    }
  } else if (lr$cost=="MAP"){
    n <- ncol(lr$X)
    for (i in 1:lr$niter){
      D_k <- diag(drop(sigmoid(lr$X%*%beta)*(1 - sigmoid(lr$X%*%beta))))
      d_k <- solve(
        lr$sigmab^2*t(lr$X)%*%D_k%*%lr$X + diag(n),
        lr$alpha*(lr$sigmab^2*t(lr$X)%*%(lr$y - sigmoid(lr$X %*% beta)) - beta)
      )
      beta <- beta + d_k
    }
  }
  return(beta)
}


#' Predicts labels of a test set using a 0.5 cutoff
#'
#' If \eqn{\sigma(x_i^\top \beta) > 0.5} then observation \eqn{i} is assigned
#' class 1. Otherwise it is assigned class 0.
#'
#' @param lr Instance of class \code{\link{logistic_regression}}.
#' @param xtest Matrix of dimension (number of test observations, number of features + 1) containing a
#' testing observation in every row and a feature in every column. Notice that the first column must contain
#' only 1s to fit the intercept.
#'
#' @return Column vector of dimension (number of test observations, 1) with a label of `0` or `1` for every
#' test observation.
#' @export
#'
#' @examples
predict.logistic_regression <- function(lr, xtest){
  # sigmoid gives probability of being in class 1. So will give (rounded) 1 to 1
  # Should be enough to just check if -xtest %*% beta > 0 for a 1/2 cutoff.
  return(round(1.0 / (1.0 + exp(-xtest %*% lr$beta))))
}
