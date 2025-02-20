#using Aggregations
using Distances
using DifferentialEquations
using NonlinearSolve, SteadyStateDiffEq, OrdinaryDiffEq
using LinearAlgebra
using Plots
<<<<<<< HEAD
=======

>>>>>>> fb6e392c1e971328317c189617dbb725d59884d6

include("functions.jl")

### Parameters 
N = 50

### Kernel Parameters
λ = 0.01
α = 25

p = (α, λ, N)
 
### Initial condition
u0 = vec(2 * rand(3, N) .- 1)

### How bad is the initial condition?
# norm(u, Inf)

ff_hessian = ODEFunction(g!; jac=h!)
ff = ODEFunction(g!)


type = eltype(u0[1])
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
@time u = solve(SteadyStateProblem(ff_hessian, u0[:], p), DynamicSS(TRBDF2()))

#### How close are we to the steady state?
print(norm(g(u,p), Inf))

### Plots
<<<<<<< HEAD
#u = reshape(u,  N,3)
#scatter(u[:,1], u[:,2], u[:,3], m=(3, 0.8, :blues, stroke(0)))


# Define a range of alpha values to test
alpha_values = 30:5:100
lambda_values = [0.01]

for λ in lambda_values
    for α in alpha_values
        println("Solving for α = $α, λ = $λ")
    
    # Update parameters
        p = (α, λ, N, r, f, a, XX, YY, ZZ, YX, ZX, ZY, Xa, Ya)
    
    # Solve the system
        @time u = solve(SteadyStateProblem(ff_hessian, u[:], p), DynamicSS(TRBDF2()))

    # Compute how close we are to the steady state
        println("Norm: ", norm(g(u, p), Inf))
    
    #plot solutions
        u = reshape(u,  N,3)
        plot = scatter(u[:,1], u[:,2], u[:,3], m=(3, 0.8, :blues, stroke(0)), title = "α = $α, λ=$λ")
        display(plot)
    end
end







=======
u = reshape(u,  N,3)
scatter(u[:,1], u[:,2], u[:,3], m=(3, 0.8, :blues, stroke(0)))
>>>>>>> fb6e392c1e971328317c189617dbb725d59884d6

