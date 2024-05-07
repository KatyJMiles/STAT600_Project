#include <RcppArmadillo.h>
#include <bits/stdc++.h> 
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;

namespace helper{ 
  NumericMatrix findPermutations(IntegerVector& c, int n) { 
    NumericVector perm(0);
    // Sort the given array 
    sort(c.begin(), c.end()); 
    // Find all possible permutations 
    do {
      for (int i = 0; i < n; i++) { 
        perm.push_back(c[i]); 
      } 
    } while (next_permutation(c.begin(), c.end()));
    NumericMatrix toReturn( c.size() , n , perm.begin() );
    return toReturn;
  }
  arma::vec perm_sort(arma::vec x, arma::vec y) {
    return x(arma::sort_index(y));
  }
}

//[[Rcpp::export]]
List em_norm(NumericVector y, NumericVector p, NumericVector theta, 
             bool classes = false) {
  int n = y.size();
  NumericVector w1_tot(n);
  NumericVector w2_tot(n);
  double mu1 = theta[0];
  double mu2 = theta[1];
  double p1 = p[0];
  double p2 = p[1];
  for (int i = 0; i < 50; i++) {
    // E
    NumericVector w1 = p1 * dnorm(y, mu1);
    NumericVector w2 = p2 * dnorm(y, mu2);
    
    w1_tot = w1 / (w1 + w2);
    w2_tot = w2 / (w1 + w2);
    
    //M
    mu1 = sum(w1_tot * y) / sum(w1_tot);
    mu2 = sum(w2_tot * y) / sum(w2_tot);
    p1 = sum(w1_tot)/n;
    p2 = sum(w2_tot)/n;
  }
  NumericVector p_hat = {p1, p2};
  NumericVector theta_hat = {mu1, mu2};
  if (classes) {
    return List::create(p_hat, theta_hat, w1_tot, w2_tot);
  }
  return List::create(p_hat, theta_hat);
}

//[[Rcpp::export]]
List em_norm4(NumericVector y, NumericVector p, NumericVector theta, 
              bool classes = false) {
  int n = y.size();
  NumericVector w1_tot(n);
  NumericVector w2_tot(n);
  NumericVector w3_tot(n);
  NumericVector w4_tot(n);
  double mu1 = theta[0];
  double mu2 = theta[1];
  double mu3 = theta[2];
  double mu4 = theta[3];
  double p1 = p[0];
  double p2 = p[1];
  double p3 = p[2];
  double p4 = p[3];
  for (int i = 0; i < 100; i++) {
    // E
    NumericVector w1 = p1 * dnorm(y, mu1);
    NumericVector w2 = p2 * dnorm(y, mu2);
    NumericVector w3 = p3 * dnorm(y, mu3);
    NumericVector w4 = p4 * dnorm(y, mu4);
    
    w1_tot = w1 / (w1 + w2 + w3 + w4);
    w2_tot = w2 / (w1 + w2 + w3 + w4);
    w3_tot = w3 / (w1 + w2 + w3 + w4);
    w4_tot = w4 / (w1 + w2 + w3 + w4);
    
    //M
    mu1 = sum(w1_tot * y) / sum(w1_tot);
    mu2 = sum(w2_tot * y) / sum(w2_tot);
    mu3 = sum(w3_tot * y) / sum(w3_tot);
    mu4 = sum(w4_tot * y) / sum(w4_tot);
    p1 = sum(w1_tot)/n;
    p2 = sum(w2_tot)/n;
    p3 = sum(w3_tot)/n;
    p4 = sum(w4_tot)/n;
  }
  NumericVector p_hat = {p1, p2, p3, p4};
  NumericVector theta_hat = {mu1, mu2, mu3, mu4};
  if (classes) {
    return List::create(p_hat, theta_hat, w1_tot, w2_tot, w3_tot, w4_tot);
  }
  return List::create(p_hat, theta_hat);
}

//[[Rcpp::export]]
vec complh(int n, vec theta_hat, vec p_hat, NumericVector latent_labels) {
  // Generating sample
  int m = theta_hat.size();
  NumericVector x(n);
  NumericMatrix z(m, n);
  IntegerVector classes =  seq(1, m);
  for (int i = 0; i < n; i++) {
    if (latent_labels[i] == 1) {
      z(0,i) = 1;
    } else {
      z(0,i) = 0;
    }
    if (latent_labels[i] == 2) {
      z(1,i) = 1;
    } else {
      z(1,i) = 0;
    }
    if (m > 2) {
      if (latent_labels[i] == 3) {
        z(2,i) = 1;
      } else {
        z(2,i) = 0;
      }
      if (latent_labels[i] == 4) {
        z(3,i) = 1;
      } else {
        z(3,i) = 0;
      }
    }
    
    x[i] = R::rnorm(theta_hat[latent_labels[i] - 1], 1);
  }
  // Label Adjustment using complete log likelihood
  NumericMatrix perm = helper::findPermutations(classes, m);
  double ratio_sum = -10000000;
  NumericVector ratio(n);
  int max_perm = 0;
  for (int k = 0; k < perm.nrow(); k++) {
    NumericVector p = perm(k , _ );
    for (int i = 0; i < n; i++) {
      int num = 0;
      for (int j = 0; j < m; j++) {
        double mean = theta_hat[p[j] - 1];
        double like = R::dnorm(x[i], mean, 1, 0);
        double eval = z(j,i)*log(p_hat[p[j] - 1]*like);
        num = num + eval;
      }
      ratio[i] = num;
    }
    if (sum(ratio) > ratio_sum) {
      ratio_sum = sum(ratio);
      max_perm = k;
    }
  }
  vec y = perm(max_perm, _);
  vec sorted_p = helper::perm_sort(p_hat, y);
  vec sorted_theta = helper::perm_sort(theta_hat, y);
  return join_cols(sorted_p, sorted_theta);
}

//[[Rcpp::export]]
vec distlat(int n, vec theta_hat, vec p_hat, NumericVector latent_labels) {
  // Generating sample
  int m = theta_hat.size();
  NumericVector x(n);
  NumericMatrix z(m, n);
  IntegerVector classes =  seq(1, m);
  for (int i = 0; i < n; i++) {
    if (latent_labels[i] ==1) {
      z(0,i) = 1;
    } else {
      z(0,i) = 0;
    }
    if (latent_labels[i] == 2) {
      z(1,i) = 1;
    } else {
      z(1,i) = 0;
    }
    if (m > 2) {
      if (latent_labels[i] == 3) {
        z(2,i) = 1;
      } else {
        z(2,i) = 0;
      }
      if (latent_labels[i] == 4) {
        z(3,i) = 1;
      } else {
        z(3,i) = 0;
      }
    }
    x[i] = R::rnorm(theta_hat[latent_labels[i] - 1], 1);
  }
  // Label Adjustment using complete log likelihood
  NumericMatrix perm = helper::findPermutations(classes, m);
  
  double ratio_sum = -10000000;
  NumericVector ratio(n);
  int max_perm = 0;
  for (int k = 0; k < perm.nrow(); k++) {
    NumericVector p = perm(k , _ );
    for (int i = 0; i < n; i++) {
      int num = 0;
      for (int j = 0; j < m; j++) {
        double mean = theta_hat[p[j] - 1];
        double like = R::dnorm(x[i], mean, 1, 0);
        double eval = z(j,i)*p_hat[p[j] - 1]*like;
        num = num + eval;
      }
      ratio[i] = num;
    }
    if (sum(ratio) > ratio_sum) {
      ratio_sum = sum(ratio);
      max_perm = k;
    }
  }
  vec y = perm(max_perm, _);
  vec sorted_p = helper::perm_sort(p_hat, y);
  vec sorted_theta = helper::perm_sort(theta_hat, y);
  return join_cols(sorted_p, sorted_theta);
}