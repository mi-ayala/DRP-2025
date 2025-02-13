using ForwardDiff
using Distances

include("functions.jl")


N = 3
λ = 2^(-6)
α = 2.0
p = (α, λ, N)

u = vec(2 * rand(3, N) .- 1)
e(u, p)
g(u, p)
h(u, p)


ForwardDiff.gradient(u -> e(u, p), u)
ForwardDiff.hessian(u -> e(u, p), u) 

