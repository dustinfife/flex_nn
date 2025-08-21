library(progressr)
require(flexplot)
require(tidyverse)
require(party)
library(furrr)
require(sensitivity)

# monte carlo parameters --------------------------------------------------

sample_size   = c(100, 200, 400, 800, 1600)
causal_struc  = c("multiplicative", "cond_imp", "suppression", "local_only",
                  "signal_diff", "linear", "interaction")


# non-varying parameters --------------------------------------------------
betas = c(.8, .7, .6, .5, .4, 0, 0, 0, 0, 0)
iterations = 1000

# begin loop --------------------------------------------------------------
source("tools/monte_carlo_function.R")

handlers("txtprogressbar")

with_progress({
  p = progressor(steps = iterations)
  
  results_lm = future_map_dfr(
    1:iterations,
    ~ {
      res = n_1_monte_carlo(.x, betas = betas, sample_size = sample_size, causal_struc = causal_struc)
      p()  # Tick progress bar
      res
    },
    .options = furrr_options(seed = TRUE)
  )
  flush.console()
  
})

write.csv(results_lm, file="tools/data/monte_carlo.csv")
str(results_lm)
# how similar are my shap versus fastshap
flexplot(ji_shap~ji_mshap   | causal_struc, data=results_lm, method="lm", ghost.line="red")
flexplot(cor_shap~cor_mshap | causal_struc, data=results_lm, method="lm", ghost.line="red")

## compute difference between mine and shap
d = results_lm %>%
  mutate(ji_bias  = ji_mshap - ji_shap,
         cor_bias = cor_mshap- cor_shap, 
         time_diff= shap_time - megashap_time)
d$time_diff %>% as.numeric %>% mean

flexplot(ji_bias~causal_struc, data=d)
flexplot(cor_bias~causal_struc, data=d)

head(results_lm)
plot_ready = results_lm %>%
  pivot_longer(ji_shap:cor_mshap) %>%
  mutate(name = factor(name))
  
head(results_lm)

# so what are the important factors?
flexplot(value~causal_struc | name, data=plot_ready)

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

