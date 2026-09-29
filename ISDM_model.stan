functions {
  // NNGP formulation folloiwng the NNGP case study in the Stan documentation:
    // https://mc-stan.org/learn-stan/case-studies/nngp.html
    // https://github.com/LuZhangstat/NNGP_STAN
    
  real nngp_w_lpdf(vector w_raw, real sigmasq, real phi,
                   matrix NN_dist, matrix NN_distM,
                   array[,] int NN_ind, int N, int M) {

    vector[N] V;
    vector[N] I_Aw = w_raw;
    int h;

    for (i in 2:N) {
      int dim = (i < (M + 1)) ? (i - 1) : M;

      matrix[dim, dim] iNNdistM;
      vector[dim]      iNNcorr;
      vector[dim]      v;
      row_vector[dim]  v2;

      if (dim == 1) {
        iNNdistM[1, 1] = 1.0;
      } else {
        h = 0;
        for (j in 1:(dim - 1)) {
          for (k in (j + 1):dim) {
            h += 1;
            iNNdistM[j, k] = exp(-phi * NN_distM[(i - 1), h]);
            iNNdistM[k, j] = iNNdistM[j, k];
          }
        }
        for (j in 1:dim){
          iNNdistM[j, j] = 1.0;
        }
      }

      matrix[dim, dim] iNNCholL = cholesky_decompose(iNNdistM);
      iNNcorr = to_vector(exp(-phi * NN_dist[(i - 1), 1:dim]));

      v       = mdivide_left_tri_low(iNNCholL, iNNcorr);
      V[i]    = 1.0 - dot_self(v);
      v2      = mdivide_right_tri_low(v', iNNCholL);
      I_Aw[i] -= v2 * w_raw[NN_ind[(i - 1), 1:dim]];
    }
    V[1] = 1.0;

    return -0.5 * (
      (1.0 / sigmasq) * dot_product(I_Aw, (I_Aw ./ V)) +
      sum(log(V)) +
      N * log(sigmasq)
    );
  }

  // N marginalisation
  real binomial_negbin_marginal_lpmf(int y_obs, real p_detect,
                                     real lambda, real phi_nb, int N_max) {
    vector[N_max - y_obs + 1] log_components;
    for (n in 1:(N_max - y_obs + 1)) {
      int N_val = y_obs + n - 1;
      log_components[n] = binomial_lpmf(y_obs | N_val, p_detect) +
                          neg_binomial_2_lpmf(N_val | lambda, phi_nb);
    }
    return log_sum_exp(log_components);
  }

}

data {
  int n_grids_total;
  int n_grids_obs;
  int n_dates;
  int n_obs_y;
  int n_obs_po;
  int n_land_covs;

  array[n_grids_obs] int obs_grid_idx;
  array[n_obs_y]     int site_to_grid;
  array[n_obs_y]     int date_y;
  array[n_obs_y]     int trap_type;
  array[n_obs_po]    int po_grid_idx;
  array[n_obs_po]    int date_po;

  array[n_obs_y]  int y;
  array[n_obs_po] int ones;

  matrix[n_grids_total, n_dates]     z_temp;
  matrix[n_grids_total, n_dates]     z_temp_sq;
  matrix[n_grids_total, n_dates]     z_rain;
  matrix[n_grids_total, n_dates]     z_rain_sq;
  matrix[n_grids_total, n_land_covs] z_land;
  vector[n_grids_total]              z_poi;
  vector[n_grids_total]              z_reports;

  int n_obs_temp_obs;
  int n_obs_temp_mis;
  array[n_obs_temp_obs] int ii_obs_temp_obs;
  array[n_obs_temp_mis] int ii_obs_temp_mis;
  vector[n_obs_temp_obs] z_obs_temp_obs;

  vector[n_grids_total] area_grid;
  real CONSTANT;
  int N_multiplier;

  int M;
  array[n_grids_total - 1, M] int NN_ind;
  matrix[n_grids_total - 1, M] NN_dist;
  matrix[n_grids_total - 1, (M * (M - 1) / 2)] NN_distM;
}

transformed data {
  int max_y = max(y);
  int N_max = max_y + max(2000, max_y * N_multiplier);
}

parameters {
  real beta0;
  real beta_temp;
  real beta_temp2;
  real beta_rain;
  real beta_rain2;
  vector[n_land_covs] beta_land;

  real alpha0;
  real alpha_obs_temp;
  vector[4] alpha_trap_raw;
  real<lower=0> sigma_trap;

  real<lower=0> sigma_obs_temp;
  real gamma_obs_temp;
  vector[n_obs_temp_mis] z_obs_temp_mis;

  real delta0;
  real delta_poi;
  real delta_reports;

  real<lower=1e-6> phi;

  real<lower=0> alpha_gp;
  real<lower=0> rho_gp;

  vector[n_grids_total] w_raw;
}

transformed parameters {
  matrix[n_grids_total, n_dates] lambda_base;
  matrix[n_grids_total, n_dates] lambda_thinned;
  vector[n_dates] background;

  vector[5] alpha_trap;
  vector[5] alpha_trap_raw_full;

  vector[n_obs_y] z_obs_temp;
  z_obs_temp[ii_obs_temp_obs] = z_obs_temp_obs;
  z_obs_temp[ii_obs_temp_mis] = z_obs_temp_mis;

  alpha_trap_raw_full[1:4] = alpha_trap_raw;
  alpha_trap_raw_full[5]   = -sum(alpha_trap_raw);
  alpha_trap = alpha_trap_raw_full * sigma_trap;

  vector[n_grids_total] spatial_field = alpha_gp * w_raw;

  for (g in 1:n_grids_total) {
    real p_thin = inv_logit(delta0 +
                            delta_poi     * z_poi[g] +
                            delta_reports * z_reports[g]);
    for (t in 1:n_dates) {
      real log_lambda_base = beta0 +
                             spatial_field[g] +
                             beta_temp  * z_temp[g, t] +
                             beta_temp2 * z_temp_sq[g, t] +
                             beta_rain  * z_rain[g, t] +
                             beta_rain2 * z_rain_sq[g, t] +
                             dot_product(beta_land, z_land[g, ]);

      lambda_base[g, t]    = exp(log_lambda_base);
      lambda_thinned[g, t] = lambda_base[g, t] * area_grid[g] * p_thin;
    }
  }

  for (t in 1:n_dates) {
    background[t] = sum(lambda_thinned[, t]) / n_obs_po;
  }
}

model {
  beta0      ~ normal(0, 1);
  beta_temp  ~ normal(0, 1);
  beta_temp2 ~ normal(0, 1);
  beta_rain  ~ normal(0, 1);
  beta_rain2 ~ normal(0, 1);
  beta_land  ~ normal(0, 1);

  alpha0         ~ normal(logit(0.10), 0.5);
  alpha_obs_temp ~ normal(0, 1);
  alpha_trap_raw ~ std_normal();
  sigma_trap     ~ exponential(3);

  gamma_obs_temp ~ normal(0, 1);
  sigma_obs_temp ~ exponential(1);

  for (k in 1:n_obs_temp_obs) {
    int idx = ii_obs_temp_obs[k];
    z_obs_temp_obs[k] ~ normal(
      gamma_obs_temp + z_temp[site_to_grid[idx], date_y[idx]],
      sigma_obs_temp
    );
  }

  for (k in 1:n_obs_temp_mis) {
    int idx = ii_obs_temp_mis[k];
    z_obs_temp_mis[k] ~ normal(
      gamma_obs_temp + z_temp[site_to_grid[idx], date_y[idx]],
      sigma_obs_temp
    );
  }

  delta0        ~ normal(0, 1);
  delta_poi     ~ normal(0, 1);
  delta_reports ~ normal(0, 1);

  phi ~ lognormal(0, 1);

  alpha_gp ~ normal(0, 1);
  rho_gp   ~ gamma(8, 0.5);

  w_raw ~ nngp_w(1.0, 1.0 / rho_gp,
                 NN_dist, NN_distM, NN_ind,
                 n_grids_total, M);

  for (k in 1:n_obs_y) {
    int g = site_to_grid[k];
    int t = date_y[k];
    real p_trap = inv_logit(alpha0 +
                            alpha_obs_temp      * z_obs_temp[k] +
                            alpha_trap[trap_type[k]]);
    target += binomial_negbin_marginal_lpmf(
      y[k] | p_trap, lambda_base[g, t], phi, N_max);
  }

  for (r in 1:n_obs_po) {
    int g = po_grid_idx[r];
    int t = date_po[r];
    target += log(lambda_thinned[g, t])
            - log(background[t])
            - log(CONSTANT);
  }
}

