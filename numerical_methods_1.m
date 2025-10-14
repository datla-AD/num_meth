
pkg load optim;                   % make sure the package is installed/loaded

x = linspace(0, 5, 60)'; rand("seed", 1);
true_params = [2, -0.8, 0.5];
y = true_params(1)*exp(true_params(2)*x) + true_params(3) + 0.05*randn(size(x));

model = @(p, x) p(1).*exp(p(2).*x) + p(3);
p0 = [1, -0.5, 0];
lb = [-Inf, -Inf, -Inf]; ub = [Inf, Inf, Inf];

[p_hat, ~, ~, ~, ~, ~, J] = lsqcurvefit(model, p0, x, y, lb, ub);
printf('Octave lsqcurvefit params = [%.4f  %.4f  %.4f]\n', p_hat);

% Optional: parameter covariance approximation
sigma2 = sum((y - model(p_hat, x)).^2) / (numel(y) - numel(p_hat));
covP = sigma2 * inv(J.'*J);
printf('Std. errors ~ [%g %g %g]\n', sqrt(diag(covP)));

% Plot
xx = linspace(min(x), max(x), 400)';
yy = model(p_hat, xx);
plot(x, y, 'o', xx, yy, '-', 'LineWidth', 2);
grid on; xlabel('x'); ylabel('y'); legend('data','fit');
title('Octave optim: lsqcurvefit');


