// distribution: lognormal
// truncation:   right
// doubly interval censored

data {
  int<lower = 0> N;                     // number of records
  vector<lower = 0>[N] a_minus;         // lower limit of event A
  vector<lower = 0>[N] a_plus;          // upper limit of event A
  vector<lower = 0>[N] b_minus;         // lower limit of event B
  vector<lower = 0>[N] b_plus;          // upper limit of event B
  real<lower = 0> upper_bound;          // the latest time of observation
  real<lower = 0> r;                    // the exponential growth rate
}

transformed data {
  array[0] int X_i;                           // empty array
}

parameters {
  real mu;                             // mean of the lognormal distribution
  real <lower = 0> sigma;              // standard deviation of the lognormal distribution
  vector<lower = 0, upper = 1>[N] a_window; // where time a lies in the event A window
  vector<lower = 0, upper = 1>[N] b_window; // where time b lies in the event B window
  vector<lower = 0, upper = 1>[N] a2_window; // where time a2 lies in the event 
}

transformed parameters {
  vector<lower = min(a_minus), upper = max(a_plus)>[N] a;
  vector<lower = min(a_minus), upper = max(a_plus)>[N] a2;
  vector<lower = min(b_minus), upper = max(b_plus)>[N] b;
  vector[N] ub;
  vector<lower = 0>[N] tstar;
  vector<lower = 0>[N] delay;

  b = b_minus + (b_plus - b_minus) .* b_window;
  for (n in 1:N)
    ub[n] = min([a_plus[n], b[n]]');
  a = a_minus + (ub - a_minus) .* a_window;
  a2 = a_minus + (ub - a_minus) .* a2_window;
  for (n in 1:N)
    ub[n] = min([upper_bound,a[n]+14]');
  tstar = upper_bound - a;
  delay = b - a;
}

model {
  mu ~ normal(1, 1);
  sigma ~ normal(1, 1);

  delay ~ lognormal(mu, sigma);

  for (n in 1:N){
    target += -log_sum_exp(
        {log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.05 | mu,sigma),
        log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.15 | mu,sigma),
        log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.25 | mu,sigma),
        log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.35 | mu,sigma),
        log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.45 | mu,sigma),
        log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.55 | mu,sigma),
        log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.65 | mu,sigma),
        log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.75 | mu,sigma),
        log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.85 | mu,sigma),
        log(0.1) + lognormal_lcdf(upper_bound - a_minus[n] + 0.95 | mu,sigma)}
      );
  }
}

generated quantities {
  real<lower = 0> mean_ = exp(mu + (sigma^2)/2);
  real<lower = 0> sd_ = sqrt((exp(sigma^2)-1)*exp(2*mu+sigma^2));
  real<lower = 0> limit_val_ = exp(mu + sigma*(1.959963984540));
  vector[N] log_likelihood;
  for (n in 1:N)
    log_likelihood[n] =
     lognormal_lpdf(b[n] - a[n] | mu, sigma) -
     lognormal_lcdf((upper_bound - a2[n]) | mu, sigma);
}
