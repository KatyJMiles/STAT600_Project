regularBoot = function(i, data, p, theta) {
  s = as.matrix(sample(data, length(data), replace = TRUE))
  if (length(p) > 2) {
    theta = unlist(em_norm4(s, p, theta))
  } else {
    theta = unlist(em_norm(s, p, theta))
  }
  return(theta)
}

strtBoot = function(data, classes, p, theta, iterations) {
  result = c()
  c = length(p)
  # Find the cluster proportions
  for (b in 1:iterations) {
    # Create bootstrap clusters
    fullSample = c()
    for (i in 1:c) {
      clusterSample = c()
      for (j in 1:length(classes)) {
        if (classes[j] == i) {
          clusterSample = append(clusterSample, data[j])
        }
      }
      if (length(clusterSample) > 0) {
        bootSample = sample(clusterSample, p[i]*length(data), TRUE)
        fullSample = append(fullSample, bootSample)
      }
    }
    
    # EM algorithm
    if (length(p) > 2) {
      result = rbind(result, unlist(em_norm4(fullSample, p, theta)))
    } else {
      result = rbind(result, unlist(em_norm(fullSample, p, theta)))
    }
  }
  return(result)
}

simulations = function(true_p, true_theta, n) {
  library(tidyverse)
  library(parallel)

  regList = list()
  stratList = list()
  adjList = list()
  distList = list()
  for (j in 1:100) {
    # Simulate Data
    p_binom = rbinom(n, 1, true_p[1])
    data = (p_binom)*rnorm(n, true_theta[1]) + (1 - p_binom)*rnorm(n, true_theta[2])
    
    # Get initial values from kmeans
    initial_theta = sort(kmeans(data, 2)$centers[,1])
    initial_p = sort(kmeans(data, 2)$size / length(data))
    
    
    # Perform EM
    em_result = em_norm(data, initial_p, initial_theta, classes = TRUE)
    theta_hat = em_result[[2]]
    p_hat = em_result[[1]]
    
    # Set number of iterations
    iterations = 100
    
    # Perform Stratified Bootstrap
    w2_tot = em_result[[4]]
    w1_tot = em_result[[3]]
    
    df = data.frame(w1_tot, 
                    w2_tot)
    df = df %>%
      rowwise() %>%
      mutate(Classification = ifelse(w1_tot == max(w1_tot, w2_tot), 1, 2))
    
    # Perform Non Parametric Bootstrap
    regBoot = t(matrix(unlist(mclapply(1:iterations, regularBoot, data = data, p = p_hat, theta = theta_hat)), nrow = 4))
    
    # Perform Parametric Bootstrap with Label Adjustment
    adjBoot = matrix(ncol = 4, nrow = iterations)
    for (i in 1:iterations) {
      # Take sample
      p_binom = rbinom(n, 1, p_hat[1])
      sample = (p_binom)*rnorm(n, theta_hat[1]) + (1 - p_binom)*rnorm(n,theta_hat[2])
      
      # Get estimates
      em = em_norm(sample, p_hat, theta_hat, classes = TRUE)
      
      w2_tot = em[[4]]
      w1_tot = em[[3]]
      
      df = data.frame(w1_tot, 
                      w2_tot)
      df = df %>%
        rowwise() %>%
        mutate(Classification = ifelse(w1_tot == max(w1_tot, w2_tot), 1, 2))
      
      # Label Adjustment
      adjBoot[i,] = complh(n, em[[2]], em[[1]], df$Classification)
    }
    
    # Perform Parametric Bootstrap with Label Adjustment DISTLAT
    distBoot = matrix(ncol = 4, nrow = iterations)
    for (i in 1:iterations) {
      # Take sample
      p_binom = rbinom(n, 1, p_hat[1])
      sample = (p_binom)*rnorm(n, theta_hat[1]) + (1 - p_binom)*rnorm(n,theta_hat[2])
      
      # Get estimates
      em = em_norm(sample, p_hat, theta_hat, classes = TRUE)
      
      w2_tot = em[[4]]
      w1_tot = em[[3]]
      
      df = data.frame(w1_tot, 
                      w2_tot)
      df = df %>%
        rowwise() %>%
        mutate(Classification = ifelse(w1_tot == max(w1_tot, w2_tot), 1, 2))
      
      # Label Adjustment
      distBoot[i,] = distlat(n, em[[2]], em[[1]], df$Classification)
    }
    
    sBoot = strtBoot(data, df$Classification, p_hat, theta_hat, iterations)
    regCI = c()
    adjCI = c()
    distCI = c()
    stratCI = c()
    # Get quantile CI
    for (i in 1:4) {
      regCI = rbind(regCI, quantile(regBoot[,i], c(.025,.975)))
      adjCI = rbind(adjCI, quantile(adjBoot[,i], c(.025, .975)))
      distCI = rbind(distCI, quantile(distBoot[,i], c(.025, .975)))
      stratCI = rbind(stratCI, quantile(sBoot[,i], c(.025,.975)))
    }
    regList = append(regList, list(regCI))
    adjList = append(adjList, list(adjCI))
    distList = append(distList, list(distCI))
    stratList = append(stratList, list(stratCI))
  }
  
  regCoverage = numeric(4)
  stratCoverage = numeric(4)
  adjCoverage = numeric(4)
  distCoverage = numeric(4)
  for (j in 1:4) {
    params = c(true_p, true_theta)
    coverage = numeric(iterations)
    for (i in 1:iterations) {
      CI = regList[[i]][j,]
      coverage[i] = ifelse(CI[1] < params[j] && CI[2] > params[j], 1, 0)
    }
    regCoverage[j] = sum(coverage)
    
    coverage = numeric(100)
    for (i in 1:100) {
      CI = adjList[[i]][j,]
      coverage[i] = ifelse(CI[1] < params[j] && CI[2] > params[j], 1, 0)
    }
    adjCoverage[j] = sum(coverage)
    
    coverage = numeric(100)
    for (i in 1:100) {
      CI = distList[[i]][j,]
      coverage[i] = ifelse(CI[1] < params[j] && CI[2] > params[j], 1, 0)
    }
    distCoverage[j] = sum(coverage)
    
    coverage = numeric(100)
    for (i in 1:100) {
      CI = stratList[[i]][j,]
      coverage[i] = ifelse(CI[1] < params[j] && CI[2] > params[j], 1, 0)
    }
    stratCoverage[j] = sum(coverage)
  }
  
  table = rbind(regCoverage, rbind(adjCoverage, rbind(distCoverage, stratCoverage)))
  return(table)
}

simulations4 = function(true_p, true_theta, n) {
  library(tidyverse)
  library(parallel)

  regList = list()
  stratList = list()
  adjList = list()
  distList = list()
  for (j in 1:100) {
    # Simulate Data
    p_multi = rmultinom(n, 1, true_p)
    dens = matrix(c(rnorm(n, true_theta[1]), rnorm(n, true_theta[2]), 
                    rnorm(n, true_theta[3]), rnorm(n, true_theta[4])), ncol = 4)
    
    data = p_multi[1,]*dens[1,] + p_multi[2,]*dens[2,] + 
      p_multi[3,]*dens[3,] + p_multi[4,]*dens[4,]
    
    # Get initial values from kmeans
    initial_theta = sort(kmeans(data, 4)$centers[,1])
    initial_p = sort(kmeans(data, 4)$size / length(data))
    
    # Perform EM
    em_result = em_norm4(data, initial_p, initial_theta, classes = TRUE)
    
    theta_hat = em_result[[2]]
    p_hat = em_result[[1]]
    
    # Set number of iterations
    iterations = 100
    
    # Perform Stratified Bootstrap
    w2_tot = em_result[[4]]
    w1_tot = em_result[[3]]
    w3_tot = em_result[[5]]
    w4_tot = em_result[[6]]
    
    df = data.frame(w1_tot, 
                    w2_tot, w3_tot, w4_tot)
    df = df %>%
      rowwise() %>%
      mutate(Classification = ifelse(w1_tot == max(w1_tot, w2_tot, w3_tot, w4_tot), 1, 
                                     ifelse(w2_tot == max(w1_tot, w2_tot, w3_tot, w4_tot), 2,
                                            ifelse(w3_tot == max(w1_tot, w2_tot, 
                                                                 w3_tot, w4_tot), 3, 4))))
    
    # Perform Non Parametric Bootstrap
    regBoot = t(matrix(unlist(mclapply(1:iterations, regularBoot, data = data, p = p_hat, theta = theta_hat)), nrow = 8))
    
    # Perform Parametric Bootstrap with Label Adjustment
    adjBoot = matrix(ncol = 8, nrow = iterations)
    for (i in 1:iterations) {
      # Simulate Data
      p_multi = rmultinom(n, 1, p_hat)
      dens = matrix(c(rnorm(n, theta_hat[1]), 
                      rnorm(n, theta_hat[2]),
                      rnorm(n, theta_hat[3]), 
                      rnorm(n, theta_hat[4])), ncol = 4)
      
      sample = p_multi[1,]*dens[1,] + p_multi[2,]*dens[2,] + 
        p_multi[3,]*dens[3,] + p_multi[4,]*dens[4,]
      
      # Get estimates
      em = em_norm4(sample, p_hat, theta_hat, classes = TRUE)
      
      w1_tot = em[[3]]
      w2_tot = em[[4]]
      w3_tot = em[[5]]
      w4_tot = em[[6]]
      
      df = data.frame(w1_tot, 
                      w2_tot, w3_tot, w4_tot)
      df = df %>%
        rowwise() %>%
        mutate(Classification = ifelse(w1_tot == max(w1_tot, w2_tot, w3_tot, w4_tot), 1, 
                                       ifelse(w2_tot == max(w1_tot, w2_tot, w3_tot, w4_tot), 2,
                                              ifelse(w3_tot == max(w1_tot, w2_tot, 
                                                                   w3_tot, w4_tot), 3, 4))))
      
      # Label Adjustment
      adjBoot[i,] = complh(n, em[[2]], em[[1]], df$Classification)
    }
    
    # Perform Parametric Bootstrap with Label Adjustment DISTLAT
    distBoot = matrix(ncol = 8, nrow = iterations)
    for (i in 1:iterations) {
      # Simulate Data
      p_multi = rmultinom(n, 1, p_hat)
      dens = matrix(c(rnorm(n, theta_hat[1]), 
                      rnorm(n, theta_hat[2]),
                      rnorm(n, theta_hat[3]), 
                      rnorm(n, theta_hat[4])), ncol = 4)
      
      sample = p_multi[1,]*dens[1,] + p_multi[2,]*dens[2,] + 
        p_multi[3,]*dens[3,] + p_multi[4,]*dens[4,]
      
      # Get estimates
      em = em_norm4(sample, p_hat, theta_hat, classes = TRUE)
      
      w1_tot = em[[3]]
      w2_tot = em[[4]]
      w3_tot = em[[5]]
      w4_tot = em[[6]]
      
      df = data.frame(w1_tot, 
                      w2_tot, w3_tot, w4_tot)
      
      df = df %>%
        rowwise() %>%
        mutate(Classification = ifelse(w1_tot == max(w1_tot, w2_tot, w3_tot, w4_tot), 1, 
                                       ifelse(w2_tot == max(w1_tot, w2_tot, w3_tot, w4_tot), 2,
                                              ifelse(w3_tot == max(w1_tot, w2_tot, 
                                                                   w3_tot, w4_tot), 3, 4))))
      
      # Label Adjustment
      distBoot[i,] = distlat(n, em[[2]], em[[1]], df$Classification)
    }
    
    sBoot = strtBoot(data, df$Classification, p_hat, theta_hat, iterations)
    
    regCI = c()
    adjCI = c()
    distCI = c()
    stratCI = c()
    # Get quantile CI
    for (i in 1:8) {
      regCI = rbind(regCI, quantile(regBoot[,i], c(.025,.975)))
      adjCI = rbind(adjCI, quantile(adjBoot[,i], c(.025, .975)))
      distCI = rbind(distCI, quantile(distBoot[,i], c(.025, .975)))
      stratCI = rbind(stratCI, quantile(sBoot[,i], c(.025,.975)))
    }
    regList = append(regList, list(regCI))
    adjList = append(adjList, list(adjCI))
    distList = append(distList, list(distCI))
    stratList = append(stratList, list(stratCI))
  }
  
  
  regCoverage = numeric(8)
  stratCoverage = numeric(8)
  adjCoverage = numeric(8)
  distCoverage = numeric(8)
  for (j in 1:8) {
    params = c(true_p, true_theta)
    coverage = numeric(iterations)
    for (i in 1:iterations) {
      CI = regList[[i]][j,]
      coverage[i] = ifelse(CI[1] < params[j] && CI[2] > params[j], 1, 0)
    }
    regCoverage[j] = sum(coverage)
    
    coverage = numeric(100)
    for (i in 1:100) {
      CI = adjList[[i]][j,]
      coverage[i] = ifelse(CI[1] < params[j] && CI[2] > params[j], 1, 0)
    }
    adjCoverage[j] = sum(coverage)
    
    coverage = numeric(100)
    for (i in 1:100) {
      CI = distList[[i]][j,]
      coverage[i] = ifelse(CI[1] < params[j] && CI[2] > params[j], 1, 0)
    }
    distCoverage[j] = sum(coverage)
    
    coverage = numeric(100)
    for (i in 1:100) {
      CI = stratList[[i]][j,]
      coverage[i] = ifelse(CI[1] < params[j] && CI[2] > params[j], 1, 0)
    }
    stratCoverage[j] = sum(coverage)
  }
  
  table = rbind(regCoverage, rbind(adjCoverage, rbind(distCoverage, stratCoverage)))
  return(table)
}
