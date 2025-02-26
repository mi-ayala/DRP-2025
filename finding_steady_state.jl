using Aggregations
using Distances
using DifferentialEquations
using NonlinearSolve, SteadyStateDiffEq, OrdinaryDiffEq
using LinearAlgebra
using Plots
using PlotlyJS
using Statistics


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


#### How close are we to the steady state?
#print(norm(g(u,p), Inf))

### Plots

#u = reshape(u,  N,3)
#scatter(u[:,1], u[:,2], u[:,3], m=(3, 0.8, :blues, stroke(0)))


# Define a range of alpha values to test
#alpha_values = 30:5:100
#lambda_values = [0.01]

#function calculate_weight(u)
   # return
function compute_minimum_distance(point1, vec_point)
    min_distance = 10000
    for point2 in eachrow(vec_point)
        current_distance = norm(point1 - point2)
        if current_distance < min_distance
            min_distance = current_distance
        end
    end
    return min_distance
end
    
        



function simulation(α_start, α_finish, step)
    
    @time u = solve(SteadyStateProblem(ff_hessian, u0[:], p), DynamicSS(TRBDF2()))
    for α in α_start : step : α_finish
        println("Solving for α = $α, λ = $λ")
    
    # Update parameters
        p = (α, λ, N, r, f, a, XX, YY, ZZ, YX, ZX, ZY, Xa, Ya)
    
    # Solve the system
        @time u = solve(SteadyStateProblem(ff_hessian, u[:], p), DynamicSS(TRBDF2()))

    # Compute how close we are to the steady state
        println("Norm: ", norm(g(u, p), Inf))
    
    #plot solutions
        u = reshape(u,N,3)
        #print(u[1])
        center_of_mass = [mean(u[:,1]), mean(u[:,2]), mean(u[:,3])]
        min_dist = compute_minimum_distance(center_of_mass,u)
    # Create the 3D scatter plot
        plot = scatter(u[:,1],u[:,2],u[:,3], 
        title = "α = $α, λ=$λ",
        label = "steady states")  # Title with your parameter values
        plot = scatter!(plot, [center_of_mass[1]], [center_of_mass[2]], [center_of_mass[3]], 
                        label = "center of mass")
        #plot = annotate!(3, 4, Plots.text("hello", :red, :right, 10))
        display(plot)
    end
end


simulation(30,30,5)
#print(compute_minimum_distance([1,1,1], [[1,2,3],[3,4,5],[6,7,8]]))


