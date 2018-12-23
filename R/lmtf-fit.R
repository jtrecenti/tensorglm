.tf_qr_fit <- function(X, y, tol, check = FALSE) {

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
    qr = matrix(1L, ncol = ncol(X), nrow = nrow(X)),
    coefficients = as.numeric(sess$run(beta)),
    rank = p,
    pivot = pv,
    tol = tol,
    pivoted = FALSE,
    # nao sei oq eh
    effects = zzz,
    residuals = zzz,
    qraux = pv
  )
}
