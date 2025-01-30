using Aggregations


# using DifferentialEquations
# using NonlinearSolve, SteadyStateDiffEq, OrdinaryDiffEq
# using LinearAlgebra


print("hello world")
print("hello world")
using Distances
include("functions.jl")



u = vec(2 * rand(3, N) .- 1)

N = 3
λ = 2^(-6)
α = 2.0
p = (α, λ, N)

e(u, p)




# # ff = ODEFunction(g!; jac=h_fast!)





# e(u, p)





# function get_state_2(u, N, ff)
#     for α in [2.0]

#         λ = 2^(-6)

#         type = eltype(u[1])
#         r = zeros(type, N, N)
#         a = zeros(type, N, N)
#         f = zeros(type, N, N)
#         XX = zeros(type, N, N)
#         YY = zeros(type, N, N)
#         ZZ = zeros(type, N, N)
#         YX = zeros(type, N, N)
#         ZX = zeros(type, N, N)
#         ZY = zeros(type, N, N)
#         Xa = zeros(type, N, N)
#         Ya = zeros(type, N, N)
#         p = (α, λ, N, r, f, a, XX, YY, ZZ, YX, ZX, ZY, Xa, Ya)

#         u = solve(SteadyStateProblem(ff, u[:], p), DynamicSS(TRBDF2()), abstol=1e-6, reltol=1e-6)

#         println("alpha: ", α, " lambda: ", λ, " residual: ", norm(g(u[:], p), Inf))
#         sol = u[:]


#     end

#     return u[:]
# end

# u = get_state_2(u, N, ff)


# #### Integration
# for i in 1:2
#     @time u = solve(ODEProblem(ff, u, (0.0, 30.0),p), QNDF(), save_everystep = false)
#     println("alpha: ", α, " lambda: ", λ, " norm: ", norm(g(u[end],p),Inf))
#     u = u[end]
#     if norm(g(u,p),Inf) < 1e-4
#         println(i)
#       break
#     end
#     end