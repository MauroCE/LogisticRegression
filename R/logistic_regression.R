run <- function(lr) UseMethod("run")
grad_ascent <- function(lr) UseMethod("grad_ascent")
newton_method <- function(lr) UseMethod("newton_method")

# TODO: Put this somewhere else. CBA at 3:30am
sigmoid <- function(x) 1.0 / (1.0 + exp(-x))

#' Logistic Regression S3 Object
#'
#' @param X
#' @param y
#' @param cost
#' @param method
#' @param sigmab
#' @param niter
#' @param alpha
#' @param gamma
#'
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
             alpha=alpha, gamma=gamma, costfunc=costfunc, beta=NULL)
  class(lr) <- "logistic_regression"
  return(lr)
}


#' Pretty-Print method for S3 Object `"logistic_regression"`.
#'
#' @param x
#' @param ...
#'
#' @return
#' @export
#'
#' @examples
print.logistic_regression <- function(x, ...){
    cat("S3 Object of Class logistic_regression.\n")
    cat("Cost Function:        ", x$cost, "\n")
    cat("Optimization Method:  ", x$method, "\n")
    cat("Solution:             ", x$beta, "\n")
  }


#' Performs Gradient Ascent on a `"logistic_regression"` object.
#' Should ideally be called internally, but can be called externally for
#' debugging purposes. Does not change the field `beta`.
#'
#' @param beta
#' @param niter
#' @param gamma
#' @param cost
#' @param sigmab
#'
#' @return
#' @export
#'
#' @examples
grad_ascent.logistic_regression <- function(lr){
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


#' Performs Newton Method on a `"logistic_regression"` object to find optimal parameters.
#'
#' @param lr
#'
#' @return
#' @export
#'
#' @examples
newton_method.logistic_regression <- function(lr){
  # Learning rate is suggested at 0.1. For 1.0 standard Newton method is recovered
  beta <- lr$start
  if (lr$cost=="MLE"){
    for (i in 1:lr$niter){
      D_k <- diag(drop(sigmoid(lr$X%*%beta)*(1 - sigmoid(lr$X%*%beta))))
      d_k <- solve(t(lr$X)%*%D_k %*% lr$X, alpha*t(lr$X) %*% (lr$y - sigmoid(lr$X %*% beta)))
      beta <- beta + d_k
    }
  } else if (lr$cost=="MAP"){
    n <- ncol(lr$X)
    for (i in 1:lr$niter){
      D_k <- diag(drop(sigmoid(lr$X%*%beta)*(1 - sigmoid(lr$X%*%beta))))
      d_k <- solve(
        lr$sigmab^2*t(lr$X)%*%D_k%*%lr$X - diag(n),
        lr$alpha*(lr$sigmab^2*t(lr$X)%*%(lr$y - sigmoid(lr$X %*% beta)) - beta)
      )
      beta <- beta + d_k
    }
  }
  return(beta)
}


#' Runs the optimization for logistic regression
#'
#' @param lr
#'
#' @return
#' @export
#'
#' @examples
run.logistic_regression <- function(lr){
  # Use selected method for selected cost function
  if      (lr$method=="BFGS")   lr$beta <- optim(par=lr$start, fn=lr$costfunc, method=lr$method)$par
  else if (lr$method=="GA")     lr$beta <- grad_ascent(lr$start, lr$niter, lr$gamma, lr$cost, lr$sigmab)
  else if (lr$method=="NEWTON") lr$beta <- newton_method(lr$start, lr$niter, lr$alpha, lr$cost, lr$sigmab)
  return(lr)
}
