# https://github.com/allisonhorst/palmerpenguins

install.packages("remotes")
remotes::install_github("allisonhorst/palmerpenguins")

library(palmerpenguins)
data(package = 'palmerpenguins')

head(penguins_raw)

write.csv(penguins_raw, file="penguins_raw.csv", na="", row.names=FALSE)
