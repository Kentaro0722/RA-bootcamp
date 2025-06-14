# 必要なパッケージの読み込みとインストール
packages <- c("ggplot2", "fixest", "bacondecomp")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

lapply(packages, library, character.only = TRUE)

# パネルデータの構築
data_sim <- expand.grid(
  id = 1:3,
  tt = 1:10
)

# シナリオの選択
scenarios <- list(
  early = c(3, 5),
  late = c(3, 8)
)

current_scenario <- "early" # "late" に切り替えることで別のシナリオを実行可能
selected_treatment <- scenarios[[current_scenario]]

# 処置の適用ルールを設定
data_sim$D <- with(data_sim, ifelse((id == 2 & tt >= selected_treatment[1]) | (id == 3 & tt >= selected_treatment[2]), 1, 0))

data_sim$treatment_effect <- ifelse(data_sim$id == 2, 2, ifelse(data_sim$id == 3, 4, 0))
data_sim$y <- with(data_sim, id + tt + D * treatment_effect)

# データの可視化
plot1 <- ggplot(data_sim, aes(x = tt, y = y, color = factor(id))) +
  geom_vline(xintercept = if (current_scenario == "early") c(2.5, 4.5) else c(2.5, 7.5), linetype = "dashed", color = "gray") +
  geom_line() +
  geom_point(size = 2) +
  scale_x_continuous(breaks = seq(1, 10, by = 1)) +
  labs(x = "Time", y = "Outcome", color = "Group", title = paste("Scenario: Treatment at", paste(selected_treatment, collapse = " and "))) +
  theme_minimal()

print(plot1)

# 固定効果モデル（TWFE）の推定
twfe_model <- feols(y ~ D | id + tt, data = data_sim)
print(twfe_model)

# Bacon-Goodman 分解の実行
bg_decomp <- bacon(y ~ D, data = data_sim, id_var = "id", time_var = "tt")
print(bg_decomp)

# 各比較の推定値の重み付き平均
weighted_avg <- weighted.mean(bg_decomp$estimate, bg_decomp$weight)
cat("Weighted Mean Estimate:", weighted_avg, "\n")

# Bacon-Goodman分解結果のプロット
plot2 <- ggplot(bg_decomp, aes(x = weight, y = estimate, color = type)) +
  geom_hline(yintercept = weighted_avg, linetype = "dashed", color = "red") +
  geom_point(size = 3) +
  labs(x = "Weight", y = "Estimate", color = "Comparison Type", title = paste("Bacon-Goodman Decomposition (Scenario:", paste(selected_treatment, collapse = " and "), ")")) +
  theme_minimal()

print(plot2)

# Naqvi(2022) https://github.com/asjadnaqvi/DiD を参考に作成