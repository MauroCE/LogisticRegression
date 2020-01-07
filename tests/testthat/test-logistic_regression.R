library(tidyverse)
library(MASS)

test_that("Class can be instantiated correctly", {
  # Generate some data
  n1 <- 100
  n2 <- 100
  m1 <- c(6, 6)
  m2 <- c(-1, 1)
  s1 <- matrix(c(1, 0, 0, 10), nrow=2, ncol=2)
  s2 <- matrix(c(1, 0, 0, 10), nrow=2, ncol=2)
  generate_binary_data <- function(n1, n2, m1, s1, m2, s2){
    # x1, x2 and y for both classes (both 0,1 and -1,1 will be created for convenience)
    class1 <- MASS::mvrnorm(n1, m1, s1)
    class2 <- MASS::mvrnorm(n2, m2, s2)
    y      <- c(rep(0, n1), rep(1, n2))   # {0 , 1}
    y2     <- c(rep(-1, n1), rep(1, n2))  # {-1, 1}
    # Generate dataframe
    data <- data.frame(rbind(class1, class2), y, y2)
    return(data)
  }
  data <- generate_binary_data(n1, n2, m1, s1, m2, s2)
  X <- data %>% dplyr::select(-y, -y2) %>% as.matrix
  y <- data %>% dplyr::select(y) %>% as.matrix
  # Instantiate object
  lr <- logistic_regression(X, y, cost="MAP")
  # Check for equality
  expect_equal(lr$y, y)
})
