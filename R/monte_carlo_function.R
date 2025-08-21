n_1_monte_carlo = function(i, betas, sample_size, causal_struc) {
  es  = runif(1, 0, .6)
  rho = runif(1, 0, .9)
  n   = sample(sample_size, size=1)
  cs  = sample(causal_struc, size=1)
  
  cor = matrix(.2, nrow=10, ncol=10)
  diag(cor) = 1
  cor[1,2] = cor[2,1] = rho
  
  d = MASS::mvrnorm(n, rep(0, 10), cor) %>%
    as.data.frame() %>%
    setNames(paste0("x", 1:10))
  
  # generate f(x)
  fx = switch(cs,
              linear         = as.vector(t(betas) %*% t(d)),
              multiplicative = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)])) + d$x1*d$x2,
              cond_imp       = {
                fx = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)]))
                ifelse(d$x1 > 0, d$x2 + fx, fx)
              },
              suppression    = {
                fx = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)]))
                ifelse(d$x1 > 0, fx, d$x2 + fx)
              },
              local_only     = {
                fx = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)]))
                ifelse(d$x1 > 2 & d$x2 < -2, 10 + fx, fx)
              },
              {
                fx = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)]))
                d$x1 = d$x1 + rnorm(n, 0, .1)
                fx + (d$x1 - d$x2)
              }
  )
  
  sigma_sq = var(fx) * (1 - es) / es
  residuals = rnorm(n, 0, sqrt(sigma_sq))
  d$y = fx + residuals
  
  lm_mod = lm(y ~ ., data=d)
  
  t1 = Sys.time()
  shap_vals = fastshap::explain(lm_mod, X=d[-ncol(d)], nsim=50, exact=FALSE, pred_wrapper=predict.lm) %>% 
    abs() %>% colMeans()
  t2 = Sys.time()
  mega_shap = megashap(lm_mod, X=d[-ncol(d)], nsim=50, predict_function=predict.lm)
  t3 = Sys.time()
  
  true_top = paste0("x", 1:5)
  shap_ranks = rank(-shap_vals)
  mega_ranks = rank(-mega_shap)
  lm_ranks = rank(-coef(lm_mod)[-1])
  
  tibble(
    i = i,
    effect_size = es,
    rho_predictors = rho,
    n = n,
    causal_struc = cs,
    shap_time = t2 - t1,
    megashap_time = t3 - t2,
    shap_spearman = cor(shap_ranks[true_top], 1:5, method="kendall"),
    mshap_spearman = cor(mega_ranks[true_top], 1:5, method="kendall"),
    coefficients   = cor(lm_ranks[true_top], 1:5, method="kendall")
  )
}
