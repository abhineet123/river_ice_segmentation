function results = vl_test_pegasos(varargin)
% VL_TEST_KDTREE
vl_test_init ;

function s = setup()
randn('state',0) ;

s.biasMultiplier = 10 ;
s.lambda = 0.01 ;

Np = 10 ;
Nn = 10 ;
Xp = diag([1 3])*randn(2, Np) ;
Xn = diag([1 3])*randn(2, Nn) ;
Xp(1,:) = Xp(1,:) + 2 + 1 ;
Xn(1,:) = Xn(1,:) - 2 + 1 ;

s.X = [Xp Xn] ;
s.y = [ones(1,Np) -ones(1,Nn)] ;
%s.w = exact_solver(s.X, s.y, s.lambda, s.biasMultiplier)
s.w = [1.181106685845652 ;
       0.098478251033487 ;
       -0.154057992404545 ] ;

function test_problem_1(s)
for conv = {@single,@double}
  vl_twister('state',0) ;
  conv = conv{1} ;
  w = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'NumIterations', 100000, ...
                 'BiasMultiplier', s.biasMultiplier, ...
                 'Preconditioner', conv([1 1 .1])) ;
  vl_assert_almost_equal(w, conv(s.w), 0.1) ;
end

function test_continue_training(s)
for conv = {@single,@double}
  conv = conv{1} ;

  vl_twister('state',0) ;
  w = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'NumIterations', 3000, ...
                 'BiasMultiplier', s.biasMultiplier) ;

  vl_twister('state',0) ;
  w1 = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'StartingIteration', 1, ...
                 'NumIterations', 1500, ...
                  'BiasMultiplier', s.biasMultiplier) ;
  w2 = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                  'StartingIteration', 1501, ...
                  'StartingModel', w1, ...
                  'NumIterations', 1500, ...
                  'BiasMultiplier', s.biasMultiplier) ;
  vl_assert_almost_equal(w,w2,1e-7) ;
end

function test_continue_training_with_perm(s)
perm = uint32(randperm(size(s.X,2))) ;
for conv = {@single,@double}
  conv = conv{1} ;

  vl_twister('state',0) ;
  w = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'NumIterations', 3000, ...
                 'BiasMultiplier', s.biasMultiplier, ...
                 'Permutation', perm) ;

  vl_twister('state',0) ;
  w1 = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                 'StartingIteration', 1, ...
                 'NumIterations', 1500, ...
                 'BiasMultiplier', s.biasMultiplier, ...
                  'Permutation', perm) ;
  w2 = vl_pegasos(conv(s.X), int8(s.y), s.lambda, ...
                  'StartingIteration', 1501, ...
                  'StartingModel', w1, ...
                  'NumIterations', 1500, ...
                  'BiasMultiplier', s.biasMultiplier, ...
                  'Permutation', perm) ;
  vl_assert_almost_equal(w,w2,1e-7) ;
end


function w = exact_solver(X, y, lambda, biasMultiplier)
N = size(X,2) ;
model = svmtrain(y', [(1:N)' X'*X], sprintf(' -c %f -t 4 ', 1/(lambda*N))) ;
w = X(:,model.SVs) * model.sv_coef ;
w(3) = - model.rho / biasMultiplier ;
format long ;
disp('model w:')
disp(w)
