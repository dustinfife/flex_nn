require(flexplot)
require(tidyverse)
require(party)

# monte carlo parameters --------------------------------------------------

effect_size   = c(.1, .3, .6)
rho_predictors= c(0, .5, .9)
n             = c(100, 200, 400, 800, 1600)
causal_struc  = c("additive", "interaction", "nonlinear")


# non-varying parameters --------------------------------------------------
betas = c(.8, .7, .6, .5, .4, 0, 0, 0, 0, 0)
nonlinear_coef = .5

# set up matrix -----------------------------------------------------------

results_lm = expand.grid(effect_size = effect_size, 
                       rho_predictors = rho_predictors,
                       n = n, 
                       causal_struc = causal_struc,
                       iterations = 1:2)
results_rf = results_nn = results_lm

nrow(results_lm)

# begin loop --------------------------------------------------------------
i = 1
for (i in 1:nrow(results_lm)) {
  
  es = results_lm$effect_size[i]    = runif(1, 0, .6)
  rho= results_lm$rho_predictors[i] = runif(1, 0, .9)
  n  = results_lm$n[i]
  cs = results_lm$causal_struc[i]
  
  # set up predictor correlation matrix
  cor = matrix(.2, nrow=10, ncol=10)
  diag(cor) = 1
  
  # replace the top two predictors' correlation
  cor[1,2] = cor[2,1] = rho
  
  # generate observed data
  d = MASS::mvrnorm(n, rep(0, times=10), cor) %>%
    data.frame %>%
    setNames(paste0("x", 1:10))

  # generate f(x)
  fx = as.vector(t(betas) %*% t(d))
  
  if (cs=="interaction") {
    fx = fx + nonlinear_coef*d$x1*d$x2
  }
  
  # compute variance of fx
  var_fitted = var(fx)
  sigma_sq = var_fitted*(1-es)/es
  residuals = rnorm(n,0,sqrt(sigma_sq))
  
  # generate y
  if (cs=="nonlinear") {
    d$y = 10/(1 + exp(-fx)) + residuals  
  } else {
    d$y = fx + residuals
  }
  
  # fit the models
  linear_model =      lm(y~., data=d)
  summary(linear_model)
  #rf_mod       = cforest(y~., data=d)
  #nn_mod       =      nn(y~., data=d)
  
  # compute fastshap values
  X = d %>% select(-y)
  pfun = make_predict_fun(X)
  time1 = Sys.time()
  shap_vals = fastshap::explain(linear_model, X=d %>% select(-y), nsim=50, exact=FALSE, pred_wrapper=predict.lm) %>% 
    abs %>% colMeans
  time2 = Sys.time()
  mega_shap =          megashap(linear_model, X=d %>% select(-y), nsim=50,  predict_function=predict.lm)
  time3 = Sys.time()
  results_lm$shap_time[i]     = time2-time1
  results_lm$megashap_time[i] = time3-time2
  
  # compare rank order
  true_top = paste0("x", 1:5)
  shap_ranks = rank(-shap_vals)  # rank highest as 1
  mshap_ranks = rank(-mega_shap)
  lm_ranks    = rank(-coef(linear_model)[-1])
  cor(shap_ranks[true_top], 1:5, method = "kendall")
  
   results_lm$shap_spearman[i] = cor(shap_ranks[true_top], 1:5, method="kendall")
  results_lm$mshap_spearman[i] = cor(mshap_ranks[true_top], 1:5, method="kendall")
  results_lm$coefficients[i]   = cor(lm_ranks[true_top], 1:5, method="kendall")
  
  results_lm$rsq[i] = summary(linear_model)$adj.r.squared
  
  message(paste0("Iteration ", i, " of ", nrow(results_lm)))
  
}

plot_ready = results_lm %>%
  pivot_longer(shap_spearman:mshap_spearman) %>%
  mutate(name = factor(name))
  
results_lm %>%
group_by(effect_size, rho_predictors, n, causal_struc) %>%
  summarize(across(shap_time:coefficients, .fns=mean))

flexplot(shap_spearman~mshap_spearman, data=results_lm)
# my shap values seem to be every bit as good as the original

rfmod = cforest(value~effect_size + rho_predictors + n +
                  causal_struc + name, data=plot_ready)

estimates(rfmod)


flexplot(value~effect_size, data=plot_ready)
# the more 'signal' there is, the better the shap values are
flexplot(value~rho_predictors, data=plot_ready)
# rho doesn't seem to make a difference
flexplot(value~effect_size | rho_predictors, data=plot_ready, ghost.line="red")
# nor does it make a difference here
flexplot(value~causal_struc, data=plot_ready)
# slightly less bias for complex structure
flexplot(value~n, data=plot_ready)
# not much going on there



flexplot(value~effect_size + causal_struc | n + rho_predictors, data=plot_ready, method="quadratic")
  
  
head(results_lm)
mean(results_lm$megashap_time)
mean(results_lm$shap_time)

mean(results_lm$shap_spearman)
mean(results_lm$mshap_spearman)

# Creates a prediction wrapper that retains column means
# make_predict_fun = function(training_data) {
#   col_means = colMeans(training_data)
#   # 
#   # function(object, newdata) {
#   #   if (!is.data.frame(newdata)) newdata = as.data.frame(newdata)
#   #   
#   #   missing_cols = setdiff(names(col_means), names(newdata))
#   #   for (col in missing_cols) {
#   #     newdata[[col]] = col_means[[col]]
#   #   }
#   #   
#   #   # Ensure correct order
#   #   newdata = newdata[, names(col_means)]
#   #   
#   #   as.numeric(predict(object, newdata = newdata))
#   # }
# }



test_row = d %>% select(-y) %>% slice(1)
pfun(linear_model, test_row)
pfun(linear_model, test_row["x1"])

