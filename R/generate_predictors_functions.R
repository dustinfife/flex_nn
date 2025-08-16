# --- helpers for Keras path ---
.mm_from_df = function(df, var_names, x_means, x_sds, factor_levels) {
  df = as.data.frame(df)
  
  # de-order factors
  df[] = lapply(df, function(x) if (is.ordered(x)) factor(x, ordered = FALSE) else x)
  
  # enforce training-time factor levels (prevents 1-level contrast errors)
  if (!is.null(factor_levels)) {
    common = intersect(names(df), names(factor_levels))
    for (v in common) df[[v]] = factor(df[[v]], levels = factor_levels[[v]])
  }
  
  dummy = df
  dummy$.y = 0
  mm = model.matrix(.y ~ ., data = dummy, na.action = stats::na.pass)[, -1, drop = FALSE]
  
  # align to training var_names (add 0-cols for missing, drop extras)
  missing_cols = setdiff(var_names, colnames(mm))
  if (length(missing_cols)) {
    add = matrix(0, nrow = nrow(mm), ncol = length(missing_cols),
                 dimnames = list(NULL, missing_cols))
    mm = cbind(mm, add)
  }
  mm = mm[, var_names, drop = FALSE]
  
  # scale like training
  if (!is.null(x_means) && !is.null(x_sds)) {
    mm = scale(mm, center = x_means, scale = x_sds)
  }
  mm
}

  
  # variables = all.vars(formula, unique = FALSE)
  # outcome   = variables[1]
  # predictors = variables[-1]
  # x_variable_named = predictors[1]
  # 
  # # 1) predict on the ORIGINAL DATA (NN-safe if needed)
  # preds = NULL
  # is_keras = any(grepl("keras", class(model)))
  # 
  # if (is_keras) {
  #   var_names = attr(model, "var_names")
  #   x_means   = attr(model, "x_means")
  #   x_sds     = attr(model, "x_sds")
  #   factor_lv = attr(model, "factor_levels")
  #   
  #   if (is.null(var_names) || is.null(x_means) || is.null(x_sds)) {
  #     stop("Keras model missing training attributes var_names/x_means/x_sds.")
  #   }
  #   
  #   # drop outcome if present so it doesn't sneak into model.matrix(.y ~ .)
  #   df_pred = data
  #   if (outcome %in% names(df_pred)) df_pred[[outcome]] = NULL
  #   
  #   mm = .mm_from_df(df_pred, var_names, x_means, x_sds, factor_lv)
  #   
  #   preds = as.numeric(predict(model, mm))
  # } else {
  #   preds = as.numeric(predict(model, newdata = data))
  # }
  # 
  # # attach predictions
  # d = data
  # d$prediction = preds
  # 
  # # 2) run your binning/sequencing on the x var
  # d = get_sequence_of_target_variable(x_variable_named, d)
  # 
  # # 3) discover which vars flexplot binned (_binned columns)
  # #    (use the same formula so we mirror flexplot's binning choices)
  # k_for_bins = bin_if_theres_a_flexplot_formula(formula, data, ...)
  # binned_vars = grep("_binned$", names(k_for_bins), value = TRUE)
  # 
  # # bind those binned columns onto our working data (same row order assumed)
  # if (length(binned_vars)) {
  #   d[, binned_vars] = k_for_bins[, binned_vars, drop = FALSE]
  # }
  # 
  # # find the x_sequence column name if it exists
  # x_var_seq = grep("_sequence$", names(d), value = TRUE)
  # # plotted vars are all binned plus the focal x
  # plotted_variables = c(binned_vars, x_variable_named)
  # 
  # # 4) group by plotted vars and average predictions (LAP)
  # n = d %>%
  #   # replace x variable with its _sequence (if present)
  #   { if (length(x_var_seq)) dplyr::mutate(., !!rlang::sym(x_variable_named) := .data[[x_var_seq]]) else . } %>%
  #   { if (length(x_var_seq)) dplyr::select(., -all_of(x_var_seq)) else . } %>%
  #   dplyr::group_by(dplyr::across(all_of(plotted_variables))) %>%
  #   dplyr::summarize(prediction = mean(prediction, na.rm = TRUE), .groups = "drop")
  # 
  # n

