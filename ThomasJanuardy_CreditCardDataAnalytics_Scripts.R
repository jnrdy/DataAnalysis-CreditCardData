#Thomas Januardy 00000046001
#Group B5

#Dataset: Credit Card Dataset for Clustering

#---Libraries---
library(tidyverse)  #manipulating and visualizing data (dplyr, purrr, ggplot2, knitr...)
library(readr)      #read in csv files faster
library(cluster)    #clustering algorithms and gap statistic
library(factoextra) #visualization of clustering algorithm results
library(GGally)     #create matrix of variable plots
library(NbClust)    #clustering algorithms and identification of best K
library(caret)      #find correlated variables
library(Amelia)     #missmap: check for missing data
library(DataExplorer) #plot_missing
library(dplyr)      #function %>%
library(ggplot2)    #plotting
library(caTools)    #set seed
library(reshape2)   #melt function
library(ggforce)    #interpretation of clusters


#---Read dataset---
cc <- read.csv("B5_ThomasJanuardy_00000046001.csv", header = TRUE)
#View(cc)

#---View and summarize the structures of the data---
str(cc)
glimpse(cc)

summary(cc)

#---Remove "CUST_ID" variable (Non numeric)---
cc <- cc[-1]

#------Check for missing data------
#---check missmap---
missmap(cc)
#No missing data

#---plot missing---
plot_missing(cc)
#CREDIT_LIMIT 0.01% data missing, MINIMUM_PAYMENTS 3.5% data missing

#---Histograms plot---
plot_histogram(cc)

#---Label "TENURE" as a factor variable---
#cc$TENURE <- factor(cc$TENURE, levels = c(6,7,8,9,10,11,12),labels = c('June','July', 'August','September', 'October', 'November', 'December'),  ordered = TRUE)

#---As numeric for 3 variables---
cc$TENURE <- as.numeric(cc$TENURE)
cc$PURCHASES_TRX <- as.numeric(cc$PURCHASES_TRX)
cc$CASH_ADVANCE_TRX <- as.numeric(cc$CASH_ADVANCE_TRX)

str(cc)

#---Transform variables for clustering---
transformed_variables <- c("BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS") #numerical data

#---Data cleaning---
clustering_data <- cc %>% drop_na() #drop missing values

plot_missing(clustering_data)


#---Splitting the data (80:20)---
code <- 5 #group B5
set.seed(code) #set seed to 5

samp <- sample(nrow(clustering_data), 0.8 * nrow(clustering_data), replace = FALSE)

trainData <- clustering_data[samp, ] #training data
nrow(trainData)

testData <- clustering_data[-samp, ] #test data
nrow(testData)

#subsetting data by choose TENURE variable = 6 / 12
#subset <- trainData
#subset <- subset(trainData, TENURE ==  6 | TENURE == 12)

#str(subset)
#summary(subset)


#---Log(1)'s variables for plotting---
clustering_data_log1 <- cc %>% na.omit() %>% mutate_at(vars(transformed_variables), funs(log(1 + .))) %>% mutate_at(c(2:17), funs(c(scale(.)))) #remove any missing values in CREDIT_LIMIT and MINIMUM_PAYMENTS, add 1 to each value to avoid log(0), scale all numeric var to mean of 0 & sd = 1.

#---Histograms plot (after cleaning)---
plot_histogram(clustering_data)

#---Plots---
plots <- as.data.frame(clustering_data_log1) %>%
  gather() %>%                             # make key-value pairs
  ggplot(aes(value)) +                     # values for each variable on x-axis
  facet_wrap(~ key, scales = "free") +  
  geom_density() +                       # plot each as density
  theme(strip.text = element_text(size=5)) # shrink text size

plots                                      # print plots

#---Corr_plots---
corr_plots <- ggpairs(as.data.frame(clustering_data_log1),                        # GGally::ggpairs to make correlation plots
                      lower = list(continuous = wrap("points", 
                                                     alpha = 0.3, size=0.1), # default point size too big-shrink & change alpha
                                   combo = wrap("dot", alpha = 0.4,size=0.2)
                      )
)

corr_plots                                 # print corr_plots



#---visualization: BALANCE by TENURE---
ggplot(clustering_data, aes(clustering_data$BALANCE,  fill = clustering_data$TENURE))+
  geom_histogram(colour="black")+
  facet_wrap(~ clustering_data$TENURE)

#---visualization: BALANCE by TENURE (Boxplot)---
ggplot(clustering_data, aes(clustering_data$BALANCE, fill = clustering_data$TENURE))+
  geom_boxplot(colour="black")+
  facet_wrap(~ clustering_data$TENURE)

#---visualization: CREDIT_LIMIT by TENURE---
ggplot(clustering_data, aes(clustering_data$CREDIT_LIMIT,  fill = clustering_data$TENURE))+
  geom_histogram(colour="black")+
  facet_wrap(~ clustering_data$TENURE)

#---visualization: CREDIT_LIMIT by TENURE (Boxplot)---
ggplot(clustering_data, aes(clustering_data$CREDIT_LIMIT, fill = clustering_data$TENURE))+
  geom_boxplot(colour="black")+
  facet_wrap(~ clustering_data$TENURE)



#---Algorithm 1: Hierarchical Clustering---

#---Euclidean distance---
trainData$TENURE <- as.numeric(trainData$TENURE) #convert TENURE variable as numeric

fit_hc_clust = hclust(dist(scale(trainData), method = "euclidean"), method = "ward")


plot(fit_hc_clust, labels = FALSE, sub = "", xlab = "", ylab = "Eclidean distance")
rect.hclust(fit_hc_clust, k = 6) #dendogram, 6 clusters

#---Choose cluster by cutting them---
hc_cluster = cutree(fit_hc_clust, k = 6) #cutting the dendogram


#---PCA plot---
hc_pc <- prcomp(scale(trainData))

fviz_pca_ind(hc_pc, habillage = hc_cluster)

#---Silhoutte plot---
hc_sil = silhouette(hc_cluster, dist(scale(trainData), method = "euclidean"), lable = FALSE)

fviz_silhouette(hc_sil, print.summary = FALSE) + theme_minimal()

#It shows if an observation is associated with the right (1) or wrong (-1) cluster. The average silhouette width is quite low (0.13 on width). Many observations probably in the wrong clusters.


#---Compare table with the original---
table_hc1 <- table(hc_cluster, trainData$TENURE)
table_hc1


#dunn index


set.seed(5)
clmethods <- c("hierarchical")
intern <- clValid::clValid(trainData, nClust = 2:6, clMethods = clmethods, validation = "internal", maxiterms = nrow(trainData))
summary(intern)
# Dunn index annd Silhouette index suggest best clustering is achieved for 2 clusters.

# Greater values of Dunn index indicates better clustering.

#Connectivity	3.1290	hierarchical	2	
#Dunn	0.2436	        hierarchical	2	
#Silhouette	0.9087	  hierarchical	2


#---Algorithm 2: K-Means---

#---Scaling data---
#use only numerical data
cc_scaled <- trainData %>% dplyr::select_if(is.numeric)
str(cc_scaled)
cor(cc_scaled)

cc_scaled <- scale(x = cc_scaled, center = TRUE, scale = TRUE)
cc_scaled %>% head()

#find the optimal k of scaled "trainData"
fviz_nbclust(scale(trainData), kmeans, method = "wss", k.max = 10)
#k is about 4 to 7, took 4 for easier segmentations.


fit_km = kmeans(scale(trainData), centers = 4) #implementing kmeans

#cluster plot
fviz_cluster(fit_km, geom = "point", data = cc_scaled) + ggtitle("k = 4")

#pca plot
hc_pc <- prcomp(scale(trainData))
fviz_pca_ind(hc_pc, habillage = fit_km$cluster)

#cluster silhouette plot
hc_sil = silhouette(fit_km$cluster, dist(scale(trainData), method = "euclidean"), lable = FALSE)

fviz_silhouette(hc_sil, print.summary = FALSE) + theme_minimal()

#dunn index
clmethods2 <- c("kmeans")
intern2 <- clValid::clValid(trainData, nClust = 2:6, clMethods = clmethods2, validation = "internal")
summary(intern2)


#---interpretation---

#grouping and factoring the cluster
c = trainData

c$cluster = fit_km$cluster

c_plots = melt(c, id.var = "cluster")

c_plots$cluster = as.factor(c$cluster)


#plotting
c_plots %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(aes(fill = cluster), outlier.size = 1) +
  facet_wrap_paginate( ~ variable, scales = "free", ncol = 3, nrow = 2, page = 1) +
  labs(x = NULL, y = NULL) +
  theme_minimal()


c_plots %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(aes(fill = cluster), outlier.size = 1) +
  facet_wrap_paginate( ~ variable, scales = "free", ncol = 3, nrow = 2, page = 2) +
  labs(x = NULL, y = NULL) +
  theme_minimal()

c_plots %>%
  ggplot(aes(x = variable, y = value)) +
  geom_boxplot(aes(fill = cluster), outlier.size = 1) +
  facet_wrap_paginate( ~ variable, scales = "free", ncol = 3, nrow = 2, page = 3) +
  labs(x = NULL, y = NULL) +
  theme_minimal()
