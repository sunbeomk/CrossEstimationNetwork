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
lambda <- runif(nitems, .65, .96)    # first three blocks (negative/positive)
gamma <- runif(nitems, -1, 1)
Phi <- diag(ntraits)
lambda[c(2, 5, 8)] <- -lambda[c(2, 5, 8)]
reverse_id <- as.integer(lambda < 0)
write.csv(reverse_id, "data/reverse_id.csv", row.names = FALSE)
class(reverse_id)
Phi
# ipsative scoring / pay attention to the reverse coded items


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
trait_id <- apply(trait_id, 2, as.integer)
write.csv(trait_id, "data/trait_id.csv", row.names = FALSE)

data_long %>%
  group_by(itemC) %>% 
  summarize(item1 = unique(item1),
            item2 = unique(item2)) %>%
  select(item1, item2) -> item_id # which items are compared
item_id <- apply(item_id, 2, as.integer)
write.csv(item_id, "data/item_id.csv", row.names = FALSE)
item_id

data_long %>%
  group_by(itemC) %>%
  summarize(block = unique(block)) %>%
  mutate(item = 1:nitems) %>%
  select(block, item) -> block_id
block_id <- apply(block_id, 2, as.integer)
write.csv(block_id, "data/block_id.csv", row.names = FALSE)
block_id


# true latent trait
data_long %>%
  group_by(person) %>%
  filter(itemC == 3) %>%
  summarise(eta1 = eta1,
            eta2 = eta2) -> eta12
data_long %>%
  group_by(person) %>%
  filter(itemC == 1) %>%
  summarise(eta3 = eta1) -> eta3
left_join(eta12, eta3) %>%
  select(eta1, eta2, eta3) %>%
  as.matrix -> eta_mat


# ipsative score
res_vec <- data[1, ]
res_vec

trait_id
reverse_id

ipsative_score <- function(res_vec, trait_id) {
  trait_vec <- rep(NA, nrow(trait_id))
  for (i in 1:nrow(trait_id)) {
    if (!reverse_id[i]) {
      trait_vec[i] <- trait_id[i, 2 - res_vec[i]]
    } else {
      trait_vec[i] <- trait_id[i, 1 + res_vec[i]]
    }
  }
  table(trait_vec)
}

ipsative_score_mat <- t(apply(data, 1, ipsative_score, trait_id = trait_id))
cor(ipsative_score_mat, eta_mat)



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


fit_lavaan <- fit_TIRT_lavaan(data_long)
summary(fit_lavaan)
eta_lavaan <- predict(fit_lavaan)
dim(eta_lavaan)
eta_lavaan_mat <- data.frame(
  eta_lavaan[eta_lavaan$trait == "trait1", "estimate"],
  eta_lavaan[eta_lavaan$trait == "trait2", "estimate"],
  eta_lavaan[eta_lavaan$trait == "trait3", "estimate"]
)

cor(eta_mat, eta_lavaan_mat)




eta_est <- as.matrix(read.csv("results/emp_study/eta_est_cen.csv", header = FALSE))
gamma_est <- as.numeric(unlist(read.csv("results/emp_study/gamma_est_cen.csv", header = FALSE)))
lambda_est <- as.numeric(unlist(read.csv("results/emp_study/lambda_est_cen.csv", header = FALSE)))
psi_sq_est <- as.numeric(unlist(read.csv("results/emp_study/psi_sq_est_cen.csv", header = FALSE)))

apply(eta_est, 2, var)

cor(gamma_est, gamma)
plot(gamma_est, gamma)
abline(0, 1)

cor(lambda_est, lambda)
lambda_est
lambda
plot(lambda_est, lambda)
abline(0, 1)
cor(eta_mat, eta_est)

psi_sq_est


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


cor(psi_sq_est, 1 - lambda^2)


