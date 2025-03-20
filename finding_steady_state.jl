using Aggregations
using Distances
using DifferentialEquations
using NonlinearSolve, SteadyStateDiffEq, OrdinaryDiffEq
using LinearAlgebra
#using Plots
using PlotlyJS
using Statistics
#plotly()


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

function compute_maximum_distance(point1, vec_point)
    max_distance = -1
    for point2 in eachrow(vec_point)
        current_distance = norm(point1 - point2)
        if current_distance > max_distance
            max_distance = current_distance
        end
    end
    return max_distance
end

function compute_all_distance(point1, vec_point)
    list = zeros(50)
    i = 1
    for point2 in eachrow(vec_point)
        list[i] = norm(point1 -point2)
        i = i +1
    end
    return list
    
end
    
        



function simulation(α_start, α_finish, step)
    num_steps = div(α_finish - α_start, step) + 1  # Add 1 to include the last step
    min_max = zeros(Float64, num_steps, 2) 
    @time u = solve(SteadyStateProblem(ff_hessian, u0[:], p), DynamicSS(TRBDF2()))
    k = 1
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
        #println(round.(u,digits = 2))
        #distances = pairwise(Euclidean(), transpose(u), dims = 2)
        
        center_of_mass = [mean(u[:,1]), mean(u[:,2]), mean(u[:,3])]
        distances = compute_all_distance(center_of_mass,u)
        min_dist = compute_minimum_distance(center_of_mass,u)
        max_dist = compute_maximum_distance(center_of_mass,u)
        min_max[k,1] = min_dist
        min_max[k,2] = max_dist
        k = k + 1
        #print(min_dist)

    # Create the 3D scatter plot
        #plot = histogram(distances,
        #plot = scatter(u[:,1],u[:,2],u[:,3], 
        #bins = range(0,0.75,20),
        #title = "α = $α, λ=$λ",)
        #label = "steady states")  # Title with your parameter values
        #plot = scatter!(plot, [center_of_mass[1]], [center_of_mass[2]], [center_of_mass[3]], 
         #               label = "center of mass")
        #plot = annotate!(3, 4, Plots.text("hello", :red, :right, 10))
        #display(plot)
        t = range(0, stop=2, length = 100)

        p = plot([
    # First trace: Center of mass
            scatter(
                x=[center_of_mass[1]], 
                y=[center_of_mass[2]], 
                z=[center_of_mass[3]],
                mode="markers",
                marker=attr(
                    size=6,
                    color="red",  
                    opacity=8
                ),
                name="Center of Mass",
                type = "scatter3d"
            ),
    
    # Steady States
            scatter(
                x=u[:,1], 
                y=u[:,2], 
                z=u[:,3],
                mode="markers",
                marker=attr(
                    size=3,
                    color="blue", 
                    opacity=0.3
                ),
                name="Steady States",
                type = "scatter3d"
            )
        ], 
        Layout(
            title=attr(
             text="α = $α, λ = $λ",   
             x=0.90,                    # Position title to the far right
             y=0.95,                    # Position title near the top
             xanchor="right",           # Right align the title
             yanchor="top"              # Top align the title
            ),
            margin=attr(l=0, r=0, b=50, t=0)  # Adjust margins around the plot
        ))
            
        
        display(p)
        
    end

    x_values = range(α_start, stop = α_finish, step = step)
    y_min = min_max[:,1]
    y_max = min_max[:,2]

    trace1 = scatter(x=x_values, y= y_min,
                    mode="lines+markers",
                    name="line_min")
    trace2 = scatter(x=x_values, y=y_max,
                    mode="lines+markers",
                    name="line_max")
    
    #line_plot = plot([trace1,trace2])
    #display(line_plot)



    
end


simulation(20,20,5)
#print(compute_minimum_distance([1,1,1], [[1,2,3],[3,4,5],[6,7,8]]))


