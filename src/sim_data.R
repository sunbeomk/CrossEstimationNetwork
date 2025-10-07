library(thurstonianIRT)
library(tidyverse)
library(vegan)
npersons <- 500
ntraits <- 3
nitems_per_block <- 3
nblocks_per_trait <- 9
nblocks <- ntraits * nblocks_per_trait / nitems_per_block
nitems <- ntraits * nblocks_per_trait
ncomparisons <- (nitems_per_block * (nitems_per_block - 1)) / 2 * nblocks


# true parameter values (Bürkner et al., 2019)
# https://cran.r-project.org/web/packages/thurstonianIRT/vignettes/TIRT_sim_tests.html
set.seed(123)
lambda <- runif(nitems, .65, .96)
gamma <- runif(nitems, -1, 1)
Phi <- diag(ntraits)
lambda
Phi


data_long <- sim_TIRT_data(
  npersons = npersons,
  ntraits = ntraits,
  nitems_per_block = nitems_per_block,
  nblocks_per_trait = nblocks_per_trait,
  gamma = gamma,
  lambda = lambda,
  Phi = Phi
)


data <- data_long %>%
  mutate(item_name = paste0("B", block, "_C", comparison)) %>%
  select(person, item_name, response) %>%                     
  pivot_wider(
    names_from = item_name,
    values_from = response
  ) %>%
  arrange(person) %>%
  select(-person)
data <- apply(data, 2, as.integer)
write.csv(data, "data/tirt_data.csv", row.names = FALSE)


# index data
data_long %>%
  group_by(itemC) %>%
  summarize(trait1 = unique(trait1),
            trait2 = unique(trait2)) %>%
  select(trait1, trait2) -> trait_id # which traits are compared
data_long %>%
  group_by(itemC) %>% 
  summarize(item1 = unique(item1),
            item2 = unique(item2)) %>%
  select(item1, item2) -> item_id # which items are compared
write.csv(apply(trait_id, 2, as.integer), "data/trait_id.csv", row.names = FALSE)
write.csv(apply(item_id, 2, as.integer), "data/item_id.csv", row.names = FALSE)

data_long %>%
  group_by(itemC) %>%
  summarize(block = unique(block)) %>%
  mutate(item = 1:nitems) %>%
  select(block, item) -> block_id

write.csv(apply(block_id, 2, as.integer), "data/block_id.csv", row.names = FALSE)
# a <- matrix(1:50, 5)
# a
# c(t(a))

# item_id <- as.matrix(item_id)
# length(unique(c(item_id)))
# item_id_ext <- matrix(rep(t(item_id), npersons), nrow = 2)
# trait_id_ext <- matrix(rep(t(trait_id), npersons), nrow = 2)
# t(item_id_ext[, 1:27]) == item_id
# t(trait_id_ext[, 1:27]) == trait_id
# c(t(data)) # CEN_y
# length(c(t(data)))
# X_item_net <- t(data)                    # 27 x 500
# X_item_net <- array(rep(X_item_net, each = nrow(data)), 
#                     dim = c(nrow(data), ncol(data), nrow(data)))
# X_item_net <- matrix(aperm(X_item_net, c(1, 2, 3)), 
#                      nrow = nrow(data) * ncol(data), 
#                      ncol = nrow(data))
# dim(X_item_net)  # 13500 x 500

# fit_stan <- fit_TIRT_stan(data_long, chains = 1, iter = 1000, warmup = 500)
# pred <- predict(fit_stan)


data_long %>%
  group_by(person) %>%
  filter(itemC == 1) %>%
  summarise(eta1 = eta1,
            eta2 = eta2) -> eta12
data_long %>%
  group_by(person) %>%
  filter(itemC == 2) %>%
  summarise(eta3 = eta2) -> eta3
left_join(eta12, eta3) %>%
  select(eta1, eta2, eta3) %>%
  as.matrix -> eta_mat


eta_est <- as.matrix(read.csv("results/emp_study/eta_est_cen.csv", header = FALSE))
gamma_est <- as.numeric(unlist(read.csv("results/emp_study/gamma_est_cen.csv", header = FALSE)))
lambda_est <- as.numeric(unlist(read.csv("results/emp_study/lambda_est_cen.csv", header = FALSE)))
psi_sq_est <- as.numeric(unlist(read.csv("results/emp_study/psi_sq_est_cen.csv", header = FALSE)))


cor(gamma_est, gamma)
cor(lambda_est, lambda)
cor(eta_mat, eta_est)


# This rotates and flips eta_est to match eta_mat as closely as possible
procrustes_solution <- procrustes(X = eta_mat, Y = eta_est)
eta_est_rotated <- procrustes_solution$Yrot
cor(eta_mat, eta_est_rotated)
var(eta_est[, 1])

plot(eta_mat[, 1], eta_est_rotated[, 1])
plot(eta_mat[, 2], eta_est_rotated[, 2])
plot(eta_mat[, 3], eta_est_rotated[, 3])

cor(rowSums(eta_mat), rowSums(eta_est_rotated))
plot(rowSums(eta_mat), rowSums(eta_est_rotated))
apply(eta_est_rotated, 2, var)



