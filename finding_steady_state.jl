using Aggregations
using Distances
using DifferentialEquations
using NonlinearSolve, SteadyStateDiffEq, OrdinaryDiffEq
using LinearAlgebra
using Plots


include("functions.jl")

### Parameters 
N = 200

### Kernel Parameters
λ = 0.2
α = 25.0

p = (α, λ, N)

### Initial condition
u = vec(2 * rand(3, N) .- 1)

### How bad is the initial condition?
# norm(u, Inf)

ff_hessian = ODEFunction(g!; jac=h!)
ff = ODEFunction(g!)


type = eltype(u[1])
r = zeros(type, N, N)
a = zeros(type, N, N)
f = zeros(type, N, N)
XX = zeros(type, N, N)
YY = zeros(type, N, N)
ZZ = zeros(type, N, N)
YX = zeros(type, N, N)
ZX = zeros(type, N, N)
ZY = zeros(type, N, N)
Xa = zeros(type, N, N)
Ya = zeros(type, N, N)
p = (α, λ, N, r, f, a, XX, YY, ZZ, YX, ZX, ZY, Xa, Ya)

### Solving
@time u = solve(SteadyStateProblem(ff_hessian, u[:], p), DynamicSS(TRBDF2()))

#### How close are we to the steady state?
norm(g(u,p), Inf)


### Plots
u = reshape(u,  N,3)
scatter(u[:,1], u[:,2], u[:,3], m=(3, 0.8, :blues, stroke(0)))

