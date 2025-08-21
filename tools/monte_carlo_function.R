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
  
  fx_function = make_fx_function(cs, betas)
  fx = fx_function(d)
  
  # Add residual noise
  sigma_sq = var(fx) * (1 - es) / es
  residuals = rnorm(n, 0, sqrt(sigma_sq))
  d$y = fx + residuals
  
  # Get "true" variable importances
  X1 = data.frame(matrix(runif(n * 10), ncol = 10))
  X2 = data.frame(matrix(runif(n * 10), ncol = 10))
  colnames(X1) = colnames(X2) = paste0("x", 1:10)
  
  sob = sobol2002(model = fx_function, X1 = X1, X2 = X2, nboot = 0)
  true_importance = sob$T %>% as.vector %>% unlist# total Sobol indices
  names(true_importance) = paste0("x", 1:10)
  
  
  lm_mod = lm(y ~ ., data=d)
  
  t1 = Sys.time()
  shap_vals = fastshap::explain(lm_mod, X=d[-ncol(d)], nsim=50, exact=FALSE, pred_wrapper=predict.lm) %>% 
    abs() %>% colMeans()
  t2 = Sys.time()
  mega_shap = megashap(lm_mod, X=d[-ncol(d)], nsim=50, predict_function=predict.lm)
  t3 = Sys.time()
  
  true_top     = order(true_importance, decreasing = TRUE)[1:5]
  shap_top     = order(shap_vals,     decreasing = TRUE)[1:5]
  mshap_top    = order(mega_shap,     decreasing = TRUE)[1:5]
  
  # compute jaccard 
  ji_shap  = length(intersect(true_top, shap_top)) / length(union(true_top, shap_top)) 
  ji_mshap = length(intersect(true_top,mshap_top)) / length(union(true_top,mshap_top)) 
  
  # compute correlation
  cor_shap  = cor(true_importance, shap_vals)
  cor_mshap = cor(true_importance, mega_shap)
  
  
  tibble(
    i = i,
    effect_size = es,
    rho_predictors = rho,
    n = n,
    causal_struc = cs,
    shap_time = t2 - t1,
    megashap_time = t3 - t2,
    ji_shap = ji_shap,
    ji_mshap = ji_mshap,
    cor_shap = cor_shap,
    cor_mshap = cor_mshap
  )
}

megashap = function(model, 
                    X, 
                    nsim = 100,
                    predict_function = predict,
                    sample_size = NULL,
                    baseline_data = NULL,
                    verbose = FALSE) {
  feature_names = colnames(X)
  p = length(feature_names)
  n = nrow(X)
  
  if (is.null(sample_size)) sample_size = n
  if (is.null(baseline_data)) {
    baseline_data = X[sample(n, sample_size, replace = TRUE), , drop = FALSE]
  }
  
  # Initialize SHAP value accumulator
  shap_values = matrix(0, nrow = n, ncol = p)
  colnames(shap_values) = feature_names
  
  for (sim in 1:nsim) {
    if (verbose && sim %% 10 == 0) message("Simulation ", sim, " of ", nsim)
    
    # Feature order to apply this iteration
    feature_order = sample(feature_names)
    
    # Bootstrap sample from X
    X_sample = as.data.frame(lapply(X, function(x) sample(x, size = sample_size, replace = TRUE)))
    
    
    # Loop over features in this order
    base_data = baseline_data[rep(1:n, each = 1), , drop = FALSE]
    preds_prev = matrix(predict_function(model, newdata = base_data), nrow = n)
    
    for (feat in feature_order) {
      # Replace the current feature with actual values from X
      base_data[[feat]] = X[[feat]]
      
      # Get new predictions with that feature replaced
      preds_new = matrix(predict_function(model, newdata = base_data), nrow = n)
      
      # SHAP contribution is the difference
      shap_values[, feat] = shap_values[, feat] + (preds_new - preds_prev)
      
      # Update prediction for next iteration
      preds_prev = preds_new
    }
  }
  
  shap_vector = structure(
    colMeans(abs(shap_values)) / nsim,
    names = colnames(shap_values)
  )
  
  
  # Sort by absolute value of SHAP
  #sorted_shap = shap_vector[order(abs(shap_vector), decreasing = TRUE)]
  shap_vector
  
}



make_fx_function = function(cs, betas) {
  force(cs); force(betas)
  
  function(d) {
    switch(cs,
           linear = as.vector(t(betas) %*% t(d)),
           
           interaction = as.vector(t(betas) %*% t(d)) + .5*d$x1*d$x2,
           
           multiplicative = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)])) + d$x1*d$x2,
           
           cond_imp = {
             fx = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)]))
             ifelse(d$x1 > 0, d$x2 + fx, fx)
           },
           
           suppression = {
             fx = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)]))
             ifelse(d$x1 > 0, fx, d$x2 + fx)
           },
           
           local_only = {
             fx = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)]))
             ifelse(d$x1 > 2 & d$x2 < -2, 10 + fx, fx)
           },
           
           # Highly Correlated Signal Difference
           {
             fx = as.vector(t(betas[-c(1:2)]) %*% t(d[,-c(1:2)]))
             d$x2 = d$x1 + rnorm(nrow(d), 0, .1)
             fx + (d$x1 - d$x2)
           }
    )
  }
}



estimate_importance_drop = function(fx_function, d) {
  var_full = var(fx_function(d))
  p = ncol(d)
  names = colnames(d)
  out = numeric(p)
  
  for (j in 1:p) {
    d_j = d
    d_j[[j]] = rnorm(nrow(d), 0, 1)  # Break structure
    fx_j = fx_function(d_j)
    var_j = var(fx_j)
    out[j] = var_full - var_j
  }
  
  names(out) = names
  out / sum(out)  # Normalize
}
