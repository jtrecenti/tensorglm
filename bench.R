rstudioapi::restartSession()
library(tensorflow)
N <- 30
p <- 2
# simulacao com .lm.fit
X <- matrix(rnorm(N * p), ncol = p)
y <- rnorm(N, apply(X, 1, sum))
system.time({
  betas_lm_fit <- .lm.fit(X, y)$coefficients
})
betas_lm_fit

aff <- .lm.fit(X, y)

aff

.tf_qr_fit <- function(X, y) {
  sess <- tf$Session()
  X_ <- tf$to_float(X)
  y_ <- tf$to_float(matrix(y, ncol = 1))
  QR <- tf$qr(X_, full_matrices = TRUE)
  q1 <- tf$slice(QR$q, shape(0L, 0L), shape(nrow(X), ncol(X)))
  q2 <- tf$slice(QR$q, shape(0L, 2L), shape(nrow(X), nrow(X)-ncol(X)))
  qy <- tf$matmul(q1, y_, transpose_a = TRUE)
  beta <- tf$matrix_solve(QR$r[seq_len(ncol(X)),], qy)
  qq1 <- sess$run(q1)
  rr1 <- sess$run(QR$r)
  qq1[upper.tri(qq1, diag = TRUE)] <- rr1[upper.tri(rr1, diag = TRUE)]
  res <- tf$matmul(tf$matmul(q2, q2, transpose_b = TRUE), y_)
  list(
    qr = qq1,
    coefficients = as.numeric(sess$run(beta)),
    rank = ncol(qq1),
    pivot = seq_len(ncol(qq1)),
    tol = 1e-07,
    pivoted = FALSE,
    residuals = as.numeric(sess$run(res)),
    # nao sei oq eh
    effects = rnorm(length(y)),
    qraux = as.numeric(qqrr$q[1,])
  )
}

all.equal(abs(qr_r), abs(aff$qr))


qqrr <- sess$run(QR)
qqrr$q %*% qqrr$r
aff$qr

qr_r <- qqrr$q
aff$qraux


# simulacao com tensorflow
library(tensorflow)
X_ <- tf$random_normal(shape(N, p))
err <- tf$random_normal(shape(N, 1L))
y_ <- tf$reshape(tf$reduce_mean(X_, 1L), shape(N, 1L)) + err
sess <- tf$Session()
invisible(sess$run(y_))

system.time({
  QR <- tf$qr(X_)
  qy <- tf$matmul(QR$q, y_, transpose_a = TRUE)
  beta <- tf$matrix_solve(QR$r, qy)
  sess$run(beta)
})
rm(list=ls())

rstudioapi::restartSession()
N <- 1e6
p <- 250
# simulacao com .lm.fit
X <- matrix(rnorm(N * p), ncol = p)
y <- rnorm(N, apply(X, 1, sum))
system.time({
  betas_lm_fit <- .lm.fit(X, y)$coefficients
})

# simulacao com tensorflow, mas passando os dados do R
library(tensorflow)
# N <- 1e6
# p <- 250
# X <- matrix(rnorm(N * p), ncol = p)
# y <- rnorm(N, apply(X, 1, sum))

system.time({
  sess <- tf$Session()
  X_ <- tf$to_float(X)
  y_ <- tf$to_float(matrix(y, ncol = 1))
  QR <- tf$qr(X_)
  qy <- tf$matmul(QR$q, y_, transpose_a = TRUE)
  beta <- tf$matrix_solve(QR$r, qy)
  betas_tensorflow <- sess$run(beta)
})

all.equal(betas_lm_fit, as.numeric(betas_tensorflow))


# --------------------------------------------
library(magrittr)
library(ggplot2)
library(tensorflow)
# library(linearmodels)

# .tf_qr_fit <- function(X, y) {
#   sess <- tf$Session()
#   X_ <- tf$to_float(X)
#   y_ <- tf$to_float(matrix(y, ncol = 1))
#   QR <- tf$qr(X_, full_matrices = TRUE)
#   q1 <- tf$slice(QR$q, shape(0L, 0L), shape(nrow(X), ncol(X)))
#   q2 <- tf$slice(QR$q, shape(0L, 2L), shape(nrow(X), nrow(X)-ncol(X)))
#   qy <- tf$matmul(q1, y_, transpose_a = TRUE)
#   beta <- tf$matrix_solve(QR$r[seq_len(ncol(X)),], qy)
#   qq1 <- sess$run(q1)
#   rr1 <- sess$run(QR$r)
#   qq1[upper.tri(qq1, diag = TRUE)] <- rr1[upper.tri(rr1, diag = TRUE)]
#   res <- tf$matmul(tf$matmul(q2, q2, transpose_b = TRUE), y_)
#   list(
#     qr = qq1,
#     coefficients = as.numeric(sess$run(beta)),
#     rank = ncol(qq1),
#     pivot = seq_len(ncol(qq1)),
#     tol = 1e-07,
#     pivoted = FALSE,
#     residuals = as.numeric(sess$run(res)),
#     # nao sei oq eh
#     effects = rnorm(length(y)),
#     qraux = as.numeric(qqrr$q[1,])
#   )
# }
.tf_qr_fit <- function(X, y) {

  sess <- tf$Session()
  X_ <- tf$to_float(X)
  y_ <- tf$to_float(matrix(y, ncol = 1))
  QR <- tf$qr(X_)

  # q1 <- tf$slice(QR$q, shape(0L, 0L), shape(nrow(X), ncol(X)))
  qy <- tf$matmul(QR$q, y_, transpose_a = TRUE)
  beta <- tf$matrix_solve(QR$r, qy)

  # rr1 <- sess$run(QR$r)
  # qq1[upper.tri(qq1, diag = TRUE)] <- rr1[upper.tri(rr1, diag = TRUE)]
  # res <- tf$matmul(tf$matmul(q2, q2, transpose_b = TRUE), y_)

  # qq1 <- sess$run(QR$q)
  zzz <- rnorm(length(y))
  p <- ncol(X)
  pv <- seq_len(p)

  list(
    qr = 2,
    coefficients = as.numeric(sess$run(beta)),
    rank = p,
    pivot = pv,
    tol = 1e-07,
    pivoted = FALSE,
    # nao sei oq eh
    effects = zzz,
    residuals = zzz,
    qraux = pv
  )
}


.dfalbel_fit <- function(X, y) {
  linear_regression <- LinearRegression$new()
  linear_regression$fit(X, matrix(y, ncol = 1))
  as.numeric(linear_regression$.__enclos_env__$private$coef)
}

simular <- function(n = 10000, p = 1000) {
  message(stringr::str_glue("Rodando com n={format(n, scientific = FALSE)}",
                            " e p={format(p, scientific = FALSE)}..."))
  X <- matrix(rnorm(n * p), ncol = p)
  y <- rnorm(n, apply(X, 1, sum))

  # simulacao com lm.fit
  time1 <- system.time({
    betas_lm_fit <- .lm.fit(X, y)$coefficients
  })

  # simulacao com tensorflow
  time2 <- system.time({
    betas_tf_fit <- .tf_qr_fit(X, y)$coefficients
  })

  time3 <- system.time({
    betas_dfalbel_fit <- .dfalbel_fit(X, y)
  })

  tibble::tibble(n = n, p = p,
                 qr.solve = time1[3],
                 tensorflow = time2[3],
                 dfalbel = time3[3])
}

res <- list(n = c(10000L),
            p = c(10L, 100L, 500L, 750L, 1000L)) %>%
  purrr::cross_df() %>%
  purrr::pmap_dfr(simular)


res %>%
  tidyr::gather(algoritmo, tempo, -n, -p) %>%
  dplyr::mutate(n = paste0("N=", format(n, scientific = FALSE,
                                        big.mark = ".",
                                        decimal.mark = ","))) %>%
  ggplot(aes(x = p, y = tempo, colour = algoritmo)) +
  geom_point() +
  geom_line() +
  facet_wrap(~n, ncol = 1, scales = "free_y") +
  theme_bw(14) +
  labs(x = "Número de variáveis",
       y = "Tempo (segundos)",
       colour = "Algoritmo")



library(tensorglm)
library(parglm)

N <- 30000
p <- 1000
X <- matrix(rnorm(N * p), ncol = p)
betas <- round(runif(p, -.1, .1), 1)
eta <- X %*% betas
# linkinv <- poisson()$linkinv
# y <- rpois(N, linkinv(eta))
y <- rnorm(N, eta)


# método atual (Fortran)
system.time({
  modelo1 <- glm(y ~ X+0, family = poisson())
})

result2 <- microbenchmark::microbenchmark(
  glm = glm(y ~ X+0, family = poisson()),
  glmtf = glm(y ~ X+0, family = poisson(), method = "glmtf.fit"),
  parglm = parglm(y ~ X+0, family = poisson(), control = parglm.control(nthreads = 4L)),
  times = 5
)

library(tidyverse)
result2 %>%
  ggplot(aes(x = expr, y = time/1000/1000/1000)) +
  geom_boxplot() +
  geom_jitter() +
  coord_flip() +
  theme_minimal()

all.equal(
  Metrics::mse(predict(glmtf), y),
  Metrics::mse(predict(parglm), y)
)




# método de ajuste novo: glmtf.fit
system.time({
  glm(y ~ X+0, family = poisson(), method = "glmtf.fit")
})

system.time({
  parglm(y ~ X+0, family = poisson(),
         control = parglm.control(nthreads = 4L))
})


all.equal(coef(modelo1), coef(modelo3))




# ------------------------------------------------------------------------------
parms <- list(
  N = c(1000, 2000),
  p = c(2, 5, 10, 100),
  family = c("Gamma", "poisson", "gaussian", "binomial")
)

gerar_dados <- function(N, p, family) {
  X <- matrix(rnorm(N * p), ncol = p)
  betas <- round(runif(p, -1, 1), 5)
  eta <- X %*% betas
  fam <- get(family, mode = "function")
  if (family == "Gamma") {
    fam <- fam("log")
  } else if (family == "poisson") {
    fam <- fam("log")
  } else {
    fam <- fam()
  }
  linkinv <- fam$linkinv
  if (family == "poisson") {
    y <- rpois(N, linkinv(eta))
  } else if (family == "gaussian") {
    y <- rnorm(N, eta)
  } else if (family == "binomial") {
    y <- rbinom(N, 1, linkinv(eta))
  } else if (family == "Gamma") {
    y <- rgamma(N, linkinv(eta) / 1, linkinv(eta) / 1) + 1e-06
  }
  list(fam = fam, X = X, beta = betas, y = y)
}

bench <- function(l, times = 3) {
  microbenchmark::microbenchmark(
    glm = glm(l$y ~ l$X+0, family = l$fam),
    glmtf = glm(l$y ~ l$X+0, family = l$fam, method = "glmtf.fit"),
    parglm = parglm(l$y ~ l$X+0, family = l$fam, control = parglm.control(nthreads = 4L)),
    times = times
  )
}
safe_bench <- possibly(bench, tibble())

dados <- parms %>%
  cross_df() %>%
  mutate(input = pmap(., gerar_dados))

pb <- progress::progress_bar$new(total = nrow(dados))
benchmarks <- dados %>%
  mutate(bench = map(input, ~{
    pb$tick()
    safe_bench(.x)
  }))

benchmarks <- fs::dir_ls("~/benchmarks/") %>%
  purrr::map_dfr(readr::read_rds, .id = "file") %>%
  tibble::as_tibble() %>%
  dplyr::mutate(nm = stringr::str_extract(file, "(?<=_)[0-9_a-z]+(?=\\.)")) %>%
  tidyr::separate(nm, c("p", "N", "family"), sep = "_", convert = TRUE)

library(ggplot2)
benchmarks %>%
  dplyr::filter(N > 100) %>%
  dplyr::group_by(N, p, family, expr) %>%
  dplyr::summarise(time = median(time)) %>%
  dplyr::ungroup() %>%
  ggplot(aes(x = p, y = time/1e9, colour = expr)) +
  geom_line() +
  geom_point() +
  scale_colour_viridis_d() +
  facet_grid(N~family) +
  theme_minimal(14)



benchmarks %>%
  # mutate(n = map_int(bench, nrow)) %>%
  # filter(n > 0) %>%
  unnest(bench) %>%
  ggplot(aes(x = p, y = time/1e9, colour = expr)) +
  geom_line() +
  geom_point() +
  scale_colour_viridis_d() +
  facet_grid(N~family) +
  theme_minimal(14)





