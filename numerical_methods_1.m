function fit_sines_demo()
  % --- 0) Setup: interval [a,b] and sample data --------------------------
  a = 0; b = 1;                      % fit on [a,b]
  N = 50;                            % number of data points
  rng(0);                            % Octave: rand("seed",0)
  x = sort(a + (b-a)*rand(N,1));     % N-by-1 column vector in [a,b]

  % True (unknown) function + noise (for demo)
  ftrue = @(t) 0.8*sin(2*pi*((t-a)/(b-a))) - 0.3*sin(5*pi*((t-a)/(b-a))) + 0.1;
  sigma = 0.08;
  y = ftrue(x) + sigma*randn(N,1);

  % --- 1) Choose sine basis size M ---------------------------------------
  % Using sines sin(j*pi*(x-a)/(b-a)) for j = 1..M
  % Note: pure sines imply f(a)=f(b)=0 for the "ideal" series on uniform grids;
  % here we do LS on possibly irregular x, so it's just a flexible basis.
  M = 12;                            % number of sine functions (columns)

  % --- 2) Build the design matrix S (N-by-M) -----------------------------
  j = 1:M;                           % 1-by-M
  theta = pi*(x - a)/(b - a);        % N-by-1
  S = sin(theta * j);                % implicit expansion: N-by-1 times 1-by-M -> N-by-M

  % --- 3) Solve for coefficients c ---------------------------------------
  % (a) Plain least-squares (preferred: backslash does QR/SVD under the hood)
  c_ls = S \ y;                      % M-by-1

  % (b) Optional: ridge-regularized solve (stabilizes if M is large)
  lambda = 1e-2;                     % try 0 (off), 1e-3, 1e-2, 1e-1, ...
  c_ridge = (S.'*S + lambda*eye(M)) \ (S.'*y);

  % --- 4) Evaluate the fit on a dense grid --------------------------------
  xfit = linspace(a,b,800).';        % fine grid
  thetaf = pi*(xfit - a)/(b - a);    % L-by-1
  Sfit = sin(thetaf * j);            % L-by-M
  yfit_ls    = Sfit * c_ls;
  yfit_ridge = Sfit * c_ridge;

  % --- 5) Report and plot -------------------------------------------------
  fprintf('M = %d sines, N = %d data\n', M, N);
  fprintf('Train MSE (LS)   : %.4g\n', mean((S*c_ls - y).^2));
  fprintf('Train MSE (Ridge): %.4g   (lambda=%.3g)\n', mean((S*c_ridge - y).^2), lambda);

  figure(1); clf; hold on; grid on;
  plot(x, y, 'ko', 'markersize', 5, 'displayname', 'data');
  plot(xfit, ftrue(xfit), 'k-', 'linewidth', 2, 'displayname', 'true f(x)');
  plot(xfit, yfit_ls, 'b-', 'linewidth', 2, 'displayname', 'LS sine fit');
  plot(xfit, yfit_ridge, 'r--', 'linewidth', 2, 'displayname', sprintf('Ridge sine fit (\\lambda=%g)', lambda));
  xlabel('x'); ylabel('y'); xlim([a b]);
  title(sprintf('Sine-series least squares on [%g,%g] (M=%d)', a, b, M));
  legend('location','best'); box on;

  % --- 6) (Optional) compare different M on the fly -----------------------
  % Try changing M to 4, 8, 16 and re-run to see under/overfitting behavior.
end


