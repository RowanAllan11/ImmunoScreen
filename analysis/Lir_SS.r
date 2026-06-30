suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2))

print(packageVersion("glmnet"))

set.seed(123)

# ------------------------------------------------------------
# 1. Read data
# ------------------------------------------------------------

df_main <- read.delim(
  "data/output/bigmhc/VR5_V3__k9/predictions_mapped.tsv",
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

immun_outputs <- read.delim(
  paste0(
    "data/output/variant_immunogenicity_scores/",
    "VR5_V3__k9/variant_immunogenicity_scores.tsv"
  ),
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# Check required columns
required_main <- c("variant_id", "VR_mutation")
required_outcome <- c("variant_id", "either_passed_count")

stopifnot(all(required_main %in% colnames(df_main)))
stopifnot(all(required_outcome %in% colnames(immun_outputs)))

# Check whether any variants have conflicting VR_mutation annotations
mutation_check <- df_main %>%
  distinct(variant_id, VR_mutation) %>%
  count(variant_id, name = "n_mutation_strings") %>%
  filter(n_mutation_strings > 1)

if (nrow(mutation_check) > 0) {
  stop(
    nrow(mutation_check),
    " variants have more than one distinct VR_mutation annotation."
  )
}

variant_df <- df_main %>%
  distinct(variant_id, VR_mutation)

cat("Number of unique variants:", nrow(variant_df), "\n")

# Complete list of variants, including WT variants
all_variants <- variant_df %>%
  select(variant_id)

# Convert semicolon-separated mutation strings to long format
mutation_long <- variant_df %>%
  separate_longer_delim(
    cols = VR_mutation,
    delim = ";"
  ) %>%
  mutate(
    VR_mutation = trimws(VR_mutation)
  ) %>%
  filter(
    !is.na(VR_mutation),
    VR_mutation != "",
    VR_mutation != "WT"
  ) %>%
  distinct(variant_id, VR_mutation) %>%
  mutate(mutation_present = 1L)

# One column per mutation
mutation_wide <- mutation_long %>%
  pivot_wider(
    id_cols = variant_id,
    names_from = VR_mutation,
    values_from = mutation_present,
    values_fill = 0L,
    values_fn = max,
    names_prefix = "mut_"
  )

# Rejoin to all variants so WT variants are retained
variant_encoded <- all_variants %>%
  left_join(mutation_wide, by = "variant_id") %>%
  mutate(
    across(
      starts_with("mut_"),
      ~ replace_na(.x, 0L)
    )
  )

stopifnot(!anyDuplicated(variant_encoded$variant_id))

cat(
  "Number of mutation predictors:",
  sum(startsWith(colnames(variant_encoded), "mut_")),
  "\n"
)

outcome_check <- immun_outputs %>%
  count(variant_id, name = "n_rows") %>%
  filter(n_rows > 1)

if (nrow(outcome_check) > 0) {
  stop(
    nrow(outcome_check),
    " variants have multiple rows in immun_outputs. ",
    "Filter to the required allele/analysis before joining."
  )
}

model_df <- variant_encoded %>%
  inner_join(
    immun_outputs %>%
      select(variant_id, either_passed_count),
    by = "variant_id"
  ) %>%
  filter(
    !is.na(either_passed_count),
    is.finite(either_passed_count),
    either_passed_count >= 0
  )

cat("Variants in model:", nrow(model_df), "\n")

summary(model_df$either_passed_count)
table(model_df$either_passed_count == 0)

mutation_columns <- grep(
  "^mut_",
  colnames(model_df),
  value = TRUE
)

mutation_counts <- colSums(
  as.matrix(model_df[, mutation_columns, drop = FALSE])
)

min_mutation_count <- 200

retained_mutations <- names(
  mutation_counts[mutation_counts >= min_mutation_count]
)

cat("Mutations before filtering:", length(mutation_columns), "\n")
cat("Mutations after filtering:", length(retained_mutations), "\n")

if (length(retained_mutations) == 0) {
  stop("No mutations passed the minimum prevalence threshold.")
}

X <- Matrix(
  as.matrix(model_df[, retained_mutations, drop = FALSE]),
  sparse = TRUE
)

y <- as.numeric(model_df$either_passed_count)

rownames(X) <- model_df$variant_id

stopifnot(nrow(X) == length(y))
stopifnot(!anyNA(X))
stopifnot(!anyNA(y))
stopifnot(all(y >= 0))
stopifnot(all(y == floor(y)))

cat("X dimensions:", nrow(X), "x", ncol(X), "\n")

set.seed(123)

cv_fit <- cv.glmnet(
  x = X,
  y = y,
  family = "gaussian",
  alpha = 1,
  standardize = FALSE,
  nfolds = 10,
  type.measure = "mse"
)

plot(cv_fit)

cat("lambda.min:", cv_fit$lambda.min, "\n")
cat("lambda.1se:", cv_fit$lambda.1se, "\n")

coef_1se <- as.matrix(
  coef(cv_fit, s = "lambda.1se")
)

selected_initial <- rownames(coef_1se)[
  coef_1se[, 1] != 0
]

selected_initial <- setdiff(
  selected_initial,
  "(Intercept)"
)

cat(
  "Mutations selected at lambda.1se:",
  length(selected_initial),
  "\n"
)

selected_initial

initial_results <- data.frame(
  mutation = rownames(coef_1se),
  coefficient = coef_1se[, 1],
  row.names = NULL
) %>%
  filter(mutation != "(Intercept)") %>%
  mutate(
    count_ratio = exp(coefficient),
    percent_change = 100 * (count_ratio - 1),
    selected = coefficient != 0
  ) %>%
  arrange(desc(abs(coefficient)))

initial_results %>%
  filter(selected) %>%
  print(n = Inf)

run_poisson_stability_selection <- function(
    X,
    y,
    lambda,
    n_repetitions = 500,
    sample_fraction = 0.5,
    seed = 123,
    standardize = FALSE
) {
  stopifnot(
    nrow(X) == length(y),
    sample_fraction > 0,
    sample_fraction < 1,
    n_repetitions >= 1
  )
  
  set.seed(seed)
  
  n <- nrow(X)
  p <- ncol(X)
  subsample_size <- floor(n * sample_fraction)
  
  selected_matrix <- matrix(
    0L,
    nrow = n_repetitions,
    ncol = p,
    dimnames = list(
      paste0("rep_", seq_len(n_repetitions)),
      colnames(X)
    )
  )
  
  coefficient_matrix <- matrix(
    0,
    nrow = n_repetitions,
    ncol = p,
    dimnames = list(
      paste0("rep_", seq_len(n_repetitions)),
      colnames(X)
    )
  )
  
  failed_fits <- integer(0)
  
  for (b in seq_len(n_repetitions)) {
    
    sampled_rows <- sample.int(
      n = n,
      size = subsample_size,
      replace = FALSE
    )
    
    fit_b <- tryCatch(
      glmnet(
        x = X[sampled_rows, , drop = FALSE],
        y = y[sampled_rows],
        family = "gaussian",
        alpha = 1,
        lambda = lambda,
        standardize = standardize,
        intercept = TRUE
      ),
      error = function(e) NULL
    )
    
    if (is.null(fit_b)) {
      failed_fits <- c(failed_fits, b)
      next
    }
    
    beta_b <- as.matrix(coef(fit_b, s = lambda))[-1, 1]
    
    selected_matrix[b, ] <- as.integer(beta_b != 0)
    coefficient_matrix[b, ] <- beta_b
    
    if (b %% 50 == 0) {
      message(
        "Completed ",
        b,
        " of ",
        n_repetitions,
        " repetitions"
      )
    }
  }
  
  successful_rows <- setdiff(
    seq_len(n_repetitions),
    failed_fits
  )
  
  if (length(successful_rows) == 0) {
    stop("Every stability-selection fit failed.")
  }
  
  selection_proportions <- colMeans(
    selected_matrix[successful_rows, , drop = FALSE]
  )
  
  mean_coefficients <- colMeans(
    coefficient_matrix[successful_rows, , drop = FALSE]
  )
  
  median_coefficients <- apply(
    coefficient_matrix[successful_rows, , drop = FALSE],
    2,
    median
  )
  
  mean_nonzero_coefficients <- vapply(
    seq_len(p),
    function(j) {
      values <- coefficient_matrix[successful_rows, j]
      values <- values[values != 0]
      
      if (length(values) == 0) {
        return(NA_real_)
      }
      
      mean(values)
    },
    numeric(1)
  )
  
  coefficient_sign_consistency <- vapply(
    seq_len(p),
    function(j) {
      values <- coefficient_matrix[successful_rows, j]
      values <- values[values != 0]
      
      if (length(values) == 0) {
        return(NA_real_)
      }
      
      max(
        mean(values > 0),
        mean(values < 0)
      )
    },
    numeric(1)
  )
  
  results <- data.frame(
    mutation = colnames(X),
    selection_proportion = selection_proportions,
    mean_coefficient = mean_coefficients,
    median_coefficient = median_coefficients,
    mean_nonzero_coefficient = mean_nonzero_coefficients,
    sign_consistency = coefficient_sign_consistency,
    mutation_count = Matrix::colSums(X),
    mutation_prevalence = Matrix::colMeans(X),
    row.names = NULL
  ) %>%
    mutate(
      mean_count_ratio = exp(mean_nonzero_coefficient),
      mean_percent_change = 100 * (
        mean_count_ratio - 1
      )
    ) %>%
    arrange(
      desc(selection_proportion),
      desc(sign_consistency)
    )
  
  list(
    results = results,
    selection_matrix = selected_matrix,
    coefficient_matrix = coefficient_matrix,
    failed_fits = failed_fits,
    lambda = lambda,
    successful_repetitions = length(successful_rows)
  )
}

t0 <- Sys.time()

stability_out <- run_poisson_stability_selection(
  X = X,
  y = y,
  lambda = cv_fit$lambda.1se,
  n_repetitions = 500,
  sample_fraction = 0.5,
  seed = 123,
  standardize = FALSE
)

t1 <- Sys.time()

print(t1 - t0)
cat(
  "Successful repetitions:",
  stability_out$successful_repetitions,
  "\n"
)

stability_threshold <- 0.80

stability_results <- stability_out$results %>%
  mutate(
    stable = selection_proportion >= stability_threshold
  )

stable_mutations <- stability_results %>%
  filter(stable) %>%
  arrange(desc(selection_proportion))

print(stable_mutations, n = Inf)

cat(
  "Number of stable mutations:",
  nrow(stable_mutations),
  "\n"
)

high_confidence_mutations <- stability_results %>%
  filter(
    selection_proportion >= 0.80,
    sign_consistency >= 0.90
  )

print(high_confidence_mutations, n = Inf)

plot_df <- stability_results %>%
  arrange(selection_proportion) %>%
  mutate(
    mutation = factor(
      mutation,
      levels = mutation
    )
  )

ggplot(
  plot_df,
  aes(
    x = mutation,
    y = selection_proportion,
    colour = stable
  )
) +
  geom_segment(
    aes(
      xend = mutation,
      y = 0,
      yend = selection_proportion
    ),
    linewidth = 0.7
  ) +
  geom_point(size = 1.8) +
  geom_hline(
    yintercept = stability_threshold,
    linetype = "dashed"
  ) +
  coord_flip() +
  scale_colour_manual(
    values = c(
      `TRUE` = "red",
      `FALSE` = "grey50"
    )
  ) +
  labs(
    x = "Mutation",
    y = "Selection proportion",
    colour = paste0(
      "Selected ≥ ",
      stability_threshold
    ),
    title = paste0(
      "Poisson LASSO stability selection: ",
      "either_passed_count"
    )
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom"
  )

top_n <- 50

top_plot_df <- stability_results %>%
  slice_max(
    order_by = selection_proportion,
    n = top_n,
    with_ties = FALSE
  ) %>%
  arrange(selection_proportion) %>%
  mutate(
    mutation = factor(
      mutation,
      levels = mutation
    )
  )

ggplot(
  top_plot_df,
  aes(
    x = mutation,
    y = selection_proportion,
    colour = selection_proportion >= stability_threshold
  )
) +
  geom_segment(
    aes(
      xend = mutation,
      y = 0,
      yend = selection_proportion
    ),
    linewidth = 0.9
  ) +
  geom_point(size = 2.2) +
  geom_hline(
    yintercept = stability_threshold,
    linetype = "dashed"
  ) +
  coord_flip() +
  scale_colour_manual(
    values = c(
      `TRUE` = "red",
      `FALSE` = "grey50"
    )
  ) +
  labs(
    x = "Mutation",
    y = "Selection proportion",
    title = "Top mutation stability-selection proportions",
    colour = NULL
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom"
  )

output_dir <- paste0(
  "data/output/stability_selection/",
  "VR5_V3__k9"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

write.table(
  stability_results,
  file = file.path(
    output_dir,
    "either_passed_count_poisson_stability_selection.tsv"
  ),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

write.table(
  stable_mutations,
  file = file.path(
    output_dir,
    "either_passed_count_stable_mutations.tsv"
  ),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

saveRDS(
  stability_out,
  file = file.path(
    output_dir,
    "either_passed_count_stability_selection.rds"
  )
)

saveRDS(
  cv_fit,
  file = file.path(
    output_dir,
    "either_passed_count_cv_glmnet.rds"
  )
)
