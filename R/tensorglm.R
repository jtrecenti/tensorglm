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
#' @param clip clip value
#'
#' @export
glmtf <- function(formula, family = gaussian, data, n = 1000, lr = 0.5,
                  clip = 1.0, scale = TRUE, verbose = FALSE) {

  tf <- tensorflow::tf

  X <- matrix(model.matrix(formula, data = data)[, -1], nrow(data))
  maxes <- 1.0
  if (scale) {
    X <- X / maxes
    maxes <- apply(X, 2, max)
    X <- t(t(X) / maxes)
  }
  X <- tf$to_float(X)
  y <- tf$to_float(matrix(eval(formula[[2]], data), nrow(X)))

  #-----------------------------------------------------------------------------
  W <- tf$Variable(tf$random_normal(tensorflow::shape(ncol(X), 1)))
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
  if (family$link == 'logit') family$linkinv <- tf$sigmoid
  mu <- family$linkinv(eta)
  eps <- 1e-08

  loss <- switch (
    family$family,
    'gaussian' = tf$reduce_mean((y - mu) ^ 2),
    'poisson' = tf$reduce_mean((y * (log((y + eps) / mu) - 1) + mu) * 2),
    'Gamma' = tf$reduce_mean((y / mu - log(y / mu) - 1) * 2),
    "binomial" = tf$reduce_mean(tf$maximum(mu, 0) - mu * y + log(1 + exp(-abs(mu))))
  )



  #-----------------------------------------------------------------------------
  optimizer <- tf$train$GradientDescentOptimizer(lr)
  gvs <- optimizer$compute_gradients(loss)
  # clipping
  invisible(
    purrr::map_if(gvs, ~!is.null(.x[[1]]), ~tf$clip_by_value(.x[[1]], -1.0, 1.0))
  )
  train <- optimizer$apply_gradients(gvs)
  sess <- tf$Session()
  sess$run(tf$global_variables_initializer())
  #-----------------------------------------------------------------------------
  for (step in seq_len(n)) {
    sess$run(train)
    if (verbose && step %% 100 == 0) {
      WW <- paste(as.character(round(unlist(sess$run(W) / maxes), 4)), collapse = ', ')
      s <- 'Iter: %d, b=%s, W=(%s)\n'
      cat(sprintf(s, step, round(sess$run(b), 4), WW))
    }
  }
  list(b = sess$run(b), W = sess$run(W) / maxes)
}

# -------------------------------------------------------------------
glmk <- function(formula, family = gaussian, data,
                 epochs = 10, lr = 0.01,
                 batch_size = nrow(data) / 10,
                 scale = TRUE, verbose = FALSE) {

  X <- matrix(model.matrix(formula, data = data), nrow(data))
  maxes <- 1.0
  if (scale) {
    X <- X / maxes
    maxes <- apply(X, 2, max)
    X <- t(t(X) / maxes)
  }
  y <- matrix(eval(formula[[2]], data), nrow(X))

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

  linkinv <- switch (
    family$link,
    "identity" = keras::k_identity,
    'log' = keras::k_exp,
    'inverse' = function(x) 1 / x,
    'logit' = keras::k_sigmoid,
  )
  eps <- 1e-08

  loss <- switch (
    family$family,
    'gaussian' = function(y, mu) keras::k_mean((y - mu) ^ 2),
    'poisson' = function(y, mu) keras::k_mean((y * (keras::k_log((y + eps) / mu) - 1) + mu) * 2),
    'Gamma' = function(y, mu) keras::k_mean((y / mu - keras::k_log(y / mu) - 1) * 2),
    "binomial" = function(y, mu) keras::k_mean(- y * keras::k_log(mu) - (1 - y) * keras::k_log(1 - mu))
  )

  X_ <- keras::layer_input(ncol(X))
  mu_ <- keras::layer_dense(X_, units = 1, use_bias = FALSE, name = "eta") %>%
    keras::layer_activation(linkinv)

  model <- keras::keras_model(X_, mu_)
  keras::compile(model, optimizer = keras::optimizer_sgd(lr), loss = loss)
  keras::fit(model, X, y, epochs = epochs, batch_size = batch_size)

  keras::get_layer(model, "eta") %>%
    keras::get_weights() %>%
    dplyr::first() %>%
    as.vector() %>%
    magrittr::divide_by(maxes)

}


glm_est <- function(formula, family = gaussian, data,
                    epochs = 10, lr = 0.01,
                    batch_size = nrow(data) / 10,
                    scale = TRUE, verbose = FALSE) {

  X <- matrix(model.matrix(formula, data = data), nrow(data))
  maxes <- 1.0
  if (scale) {
    X <- X / maxes
    maxes <- apply(X, 2, max)
    X <- t(t(X) / maxes)
  }
  y <- matrix(eval(formula[[2]], data), nrow(X))

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
