# Base R Implementation of Logistic Regression with Regularization

This is a very simple package containing base R code for Logistic Regression. It implements binary logistic regression where
the optimization is performed for two cost functions:

- Negative Log-Likelihood, corresponding to doing MLE.
- Negative Log-Likelihood plus a (ridge) regularization term. This is equivalent to do Maximum-A-Posteriori with an isotropic
  Gaussian prior. Notice that in this specific implementation **regularization is done also on the intercept**.
  
In both cases, 2 implementation methods are implemented in base R:

- Gradient Ascient (using a learning rate)
- Newton's Method (again using a learning rate)

A third optimization method is possible via a call to the `optim` function in `R`. The method is customizable, although it is
`"BFGS"` by default.
