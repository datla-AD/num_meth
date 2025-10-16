% fd_stability_demo.m
% Finite-difference stability demo with an initial-value equation.
% - PDE: u_t = alpha * u_xx on x in (0,L), u(0,t)=u(L,t)=0
% - Schemes: FTCS (explicit), BTCS (implicit), CN (Crank–Nicolson)
% - Also: scalar IVP y' = lambda y with Euler Fwd/Back to visualize stability.
%
% Run:  fd_stability_demo

function fd_stability_demo()
  close all; clc;

  % --- PDE parameters
  alpha = 1.0;         % diffusion coefficient
  L = 1.0;             % domain length
  Nx = 80;             % number of interior points
  T  = 0.20;           % final time

  % Grid / CFL helper
  dx = L/(Nx+1);
  rCFL = 0.5;          % FTCS stability requires r <= 1/2
  dt_stable   = 0.49 * dx^2/alpha;  % slightly below the limit
  dt_unstable = 0.60 * dx^2/alpha;  % above the limit -> blow-up for FTCS

  % Initial condition (compatible with Dirichlet 0 at boundaries):
  u0fun = @(x) sin(pi*x/L) + 0.25*sin(3*pi*x/L);

  printf("dx=%.4g, dt_stable=%.4g, dt_unstable=%.4g (FTCS limit = %.4g)\n", ...
         dx, dt_stable, dt_unstable, 0.5*dx^2/alpha);

  % --- FTCS: stable run
  [x,t,U_ftcs_ok] = heat1D_fd(alpha,L,Nx,T,dt_stable,"FTCS",u0fun);
  figure(1); plot_solution(x,t,U_ftcs_ok,"FTCS stable (r<1/2)");
  % growth check
  max_norm = max(max(abs(U_ftcs_ok)));
  printf("FTCS stable run: max |u| over time = %.4g\n", max_norm);

  % --- FTCS: unstable run
  [x,t,U_ftcs_bad] = heat1D_fd(alpha,L,Nx,T,dt_unstable,"FTCS",u0fun);
  figure(2); plot_solution(x,t,U_ftcs_bad,"FTCS UNSTABLE (r>1/2) — expect blow-up");
  max_norm_bad = max(max(abs(U_ftcs_bad)));
  printf("FTCS unstable run: max |u| over time = %.4g (should grow fast)\n", max_norm_bad);

  % --- BTCS (implicit): unconditionally stable
  dt_big = 3.0 * dx^2/alpha;  % ridiculous timestep that kills FTCS
  [x,t,U_btcs] = heat1D_fd(alpha,L,Nx,T,dt_big,"BTCS",u0fun);
  figure(3); plot_solution(x,t,U_btcs,"BTCS implicit (unconditionally stable; large dt)");

  % --- CN (Crank–Nicolson): A-stable, less dissipative than BTCS
  [x,t,U_cn] = heat1D_fd(alpha,L,Nx,T,dt_big,"CN",u0fun);
  figure(4); plot_solution(x,t,U_cn,"Crank–Nicolson (A-stable; large dt)");

  % --- Show spectra-based dt bound (optional, explanatory)
  N = Nx;
  A = fd_laplacian_matrix(N);      % discrete Laplacian (Dirichlet, interior)
  lam = eig(full(A));              % real, negative
  lam_min = min(lam);              % most negative eigenvalue
  dt_euler_bound = -2/(alpha*lam_min); % Forward Euler |1+z|<1  =>  -2 < z < 0
  printf("Spectral forward-Euler bound: dt <= %.4g (matches ~ dx^2/(2*alpha))\n", dt_euler_bound);

  % --- Scalar IVP stability visualization (y' = lambda y)
  lambda = -50; y0 = 1; Tode = 0.2;
  dt1 = 0.01; dt2 = 0.05;  % try a small and a large dt
  [t_fe1,y_fe1] = ivp_euler(lambda,y0,dt1,Tode,"fwd");
  [t_fe2,y_fe2] = ivp_euler(lambda,y0,dt2,Tode,"fwd");
  [t_be2,y_be2] = ivp_euler(lambda,y0,dt2,Tode,"bwd");

  figure(5);
  plot(t_fe1,y_fe1,'-o'); hold on;
  plot(t_fe2,y_fe2,'-s');
  plot(t_be2,y_be2,'-d');
  xlabel('t'); ylabel('y'); grid on;
  title(sprintf("IVP y'=\\lambda y (\\lambda=%g): Forward vs Backward Euler",lambda));
  legend(sprintf('Forward Euler dt=%.3g',dt1), ...
         sprintf('Forward Euler dt=%.3g',dt2), ...
         sprintf('Backward Euler dt=%.3g',dt2), 'location','northeast');

endfunction

% ================= CORE SOLVER =================
function [x,t,U] = heat1D_fd(alpha,L,Nx,T,dt,scheme,u0fun)
  % Builds and steps the semi-discrete system u_t = alpha * A u
  % A: 2nd-deriv central-diff Laplacian on Nx interior nodes with Dirichlet BCs.

  if nargin < 7, u0fun = @(x) sin(pi*x/L); end
  if nargin < 6, scheme = "FTCS"; end

  N = Nx;
  dx = L/(N+1);
  x = linspace(dx, L-dx, N).';    % interior points only
  A = fd_laplacian_matrix(N);     % (1/dx^2)*[1 -2 1] stencil, boundary handled by interior-only
  A = (alpha/dx^2) * A;           % scale by alpha/dx^2

  u0 = u0fun(x);
  M = ceil(T/dt);                 % number of steps
  t = linspace(0, M*dt, M+1);
  U = zeros(N, M+1);
  U(:,1) = u0;

  I = speye(N);
  r = alpha*dt/dx^2;

  switch upper(scheme)
    case "FTCS"   % Forward Euler in time
      % u^{n+1} = (I + dt*A) u^n
      S = (I + dt*A);
      for n=1:M
        U(:,n+1) = S * U(:,n);
      end

    case "BTCS"   % Backward Euler in time
      % (I - dt*A) u^{n+1} = u^n
      LHS = (I - dt*A);
      % One factorization reused:
      % For symmetric positive-definite (-A is SPD), we can use \ directly.
      for n=1:M
        U(:,n+1) = LHS \ U(:,n);
      end

    case "CN"     % Crank–Nicolson
      % (I - dt/2*A) u^{n+1} = (I + dt/2*A) u^n
      LHS = (I - 0.5*dt*A);
      RHS = (I + 0.5*dt*A);
      for n=1:M
        U(:,n+1) = LHS \ (RHS * U(:,n));
      end

    otherwise
      error("Unknown scheme: %s. Use 'FTCS', 'BTCS', or 'CN'.", scheme);
  endswitch
endfunction

% Discrete Laplacian on interior nodes with Dirichlet BCs (N x N)
function A = fd_laplacian_matrix(N)
  e = ones(N,1);
  A = spdiags([e -2*e e], [-1 0 1], N, N);
endfunction

% ================= PLOTTING HELPERS =================
function plot_solution(x,t,U,ttl)
  % space–time surface and a few time slices
  subplot(1,2,1);
  % Create a surface: x along one axis, t along the other
  [TT,XX] = meshgrid(t,x);
  surf(TT',XX',U'); shading interp; view(130,30);
  xlabel('t'); ylabel('x'); zlabel('u(x,t)'); title(ttl);
  grid on;

  subplot(1,2,2);
  hold on; grid on;
  ks = unique(round(linspace(1, size(U,2), 5))); % 5 slices
  for k = ks
    plot(x, U(:,k), 'DisplayName', sprintf('t=%.3g', t(k)));
  endfor
  xlabel('x'); ylabel('u(x,t)');
  title('Selected time slices');
  legend('location','northeastoutside');
endfunction

% ================= SCALAR IVP STABILITY (y' = lambda y) =================
function [t,y] = ivp_euler(lambda,y0,dt,T,which)
  M = ceil(T/dt);
  t = linspace(0, M*dt, M+1);
  y = zeros(1,M+1); y(1) = y0;

  switch lower(which)
    case {"fwd","forward","explicit"}
      g = @(z) (1 + z);         % amplification factor for Forward Euler
      for n=1:M
        y(n+1) = g(lambda*dt) * y(n);
      end
    case {"bwd","backward","implicit"}
      g = @(z) 1./(1 - z);      % amplification factor for Backward Euler
      for n=1:M
        y(n+1) = g(lambda*dt) * y(n);
      end
    otherwise
      error("which must be 'fwd' or 'bwd'");
  endswitch
endfunction

% ================= ENTRY POINT =================
fd_stability_demo();

