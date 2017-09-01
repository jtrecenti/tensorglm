#' Run GLM using TensorFlow
#'
#' Current implementation works for gaussian, Gamma and Poisson
#'
#' @param formula (like in glm)
#' @param family family and link (like in glm)
#' @param data data.frmae
#' @param n number of GD iterations
#' @param lr learning rate
#' @param scale divide X variables by max before ajusting?
#' @param verbose print model info
#'
#' @export
glmtf <- function(formula, family = gaussian, data, n = 1000, lr = 0.5,
                  scale = TRUE, verbose = FALSE) {
  tf <- tensorflow::tf
  X <- matrix(model.matrix(formula, data = data)[, -1], nrow(data))
  maxes <- 1
  if (scale) {
    X <- X / maxes
    maxes <- apply(X, 2, max)
    X <- t(t(X) / maxes)
  }
  X <- tf$to_float(X)
  y <- tf$to_float(matrix(eval(formula[[2]], data), nrow(X)))
  #-----------------------------------------------------------------------------
  W <- tf$Variable(tf$random_uniform(tensorflow::shape(ncol(X), 1), -1.0, 1.0))
  b <- tf$Variable(tf$zeros(tensorflow::shape(1L)))
  eta <- tf$matmul(X, W) + b
  #-----------------------------------------------------------------------------
  if (is.character(family))
    family <- get(family, mode = "function", envir = parent.frame())
  if (is.function(family))
    family <- family()
  if (is.null(family$family)) {
    print(family)
    stop("'family' not recognized")
  }
  if (family$link == 'log') family$linkinv <- exp
  mu <- family$linkinv(eta)
  eps <- 1e-08
  loss <- switch (
    family$family,
    'gaussian' = tf$reduce_mean((y - mu) ^ 2),
    'poisson' = tf$reduce_mean((y * (log((y + eps) / mu) - 1) + mu) * 2),
    'Gamma' = tf$reduce_mean((y / mu - log(y / mu) - 1) * 2)
  )
  #-----------------------------------------------------------------------------
  optimizer <- tf$train$GradientDescentOptimizer(lr, use_locking = TRUE)
  train <- optimizer$minimize(loss)
  sess <- tf$Session()
  sess$run(tf$global_variables_initializer())
  #-----------------------------------------------------------------------------
  for (step in seq_len(n)) {
    sess$run(train)
    if (verbose && step %% 100 == 0) {
      WW <- paste(as.character(round(unlist(sess$run(W) / maxes), 4)),
                  collapse = ', ')
      s <- 'Iter: %d, b=%s, W=(%s)\n'
      cat(sprintf(s, step, round(sess$run(b), 4), WW))
    }
  }
  list(b = sess$run(b), W = sess$run(W) / maxes)
}

## modelo GLM
# coef(m)
#
# g <- function(x) exp(x)
#
#
# ## POISSON
# x <- runif(1e5, min = 0, max = 1)
# y <- rpois(1e5, exp(2.5 + 0.1 * x))
#
# W <- tf$Variable(tf$random_uniform(tensorflow::shape(1L), -1.0, 1.0))
# b <- tf$Variable(tf$zeros(tensorflow::shape(1L)))
#
# mu_hat <- g(eta)
# eps <- 1e-8
# loss <- tf$reduce_mean(2 * (y * (log((y + eps) / mu_hat) - 1) + mu_hat))
#
# ## GAMMA
# x <- runif(1e5, min = 0, max = 1)
# y <- gamlss.dist::rGA(1e5, g(2.5 + 0.1 * x))
#
# W <- tf$Variable(tf$random_uniform(tensorflow::shape(1L), -1.0, 1.0))
# b <- tf$Variable(tf$zeros(tensorflow::shape(1L)))
# eta <- W * x + b
# mu_hat <- g(eta)
# loss <- tf$reduce_mean(2 * (y / mu_hat - log(y / mu_hat) - 1))
#
# ################################################################################