using ForwardDiff
using Distances
using LinearAlgebra

include("functions.jl")


N = 3
λ = 2^(-6)
α = 2.0
p = (α, λ, N)

u = vec(2 * rand(3, N) .- 1)
e(u, p)
@time g(u, p)
@time h(u, p)


@time ForwardDiff.gradient(u -> e(u, p), u)
@time ForwardDiff.hessian(u -> e(u, p), u) 


norm(ForwardDiff.hessian(u -> e(u, p), u) +  h(u, p), Inf)