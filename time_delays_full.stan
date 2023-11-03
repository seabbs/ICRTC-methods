// distribution: lognormal
// truncation:   right
// doubly interval censored

functions {
  real denom(real x,
              real xc,
              array[] real theta,
              array[] real x_r,
              array[] int x_i) {
    real mu = theta[1];
    real sigma = theta[2];
    real tstar = theta[3];

    return lognormal_cdf(tstar - x | mu,sigma);
  }
}


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
  vector<lower = 0, upper = 1>[N] a2_window; // where time a2 lies in the event A window
  vector<lower = 0>[N] t0;                  // time from O to A
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
    target += - log(
      integrate_1d(
        denom, a_minus[n], a_plus[n], {mu,sigma,upper_bound},{r},X_i)
      );
  }
}

generated quantities {
  real<lower = 0> mean_ = exp(mu + (sigma^2)/2);
  real<lower = 0> sd_ = sqrt((exp(sigma^2)-1)*exp(2*mu+sigma^2));
  real<lower = 0> limit_val_ = exp(mu + sigma*(1.959963984540));
  vector[N] log_likelihood;
  for (n in 1:N)
    log_likelihood[n] = lognormal_lpdf(b[n] - a[n] | mu, sigma) - lognormal_lcdf((upper_bound - a2[n]) | mu, sigma);
}
