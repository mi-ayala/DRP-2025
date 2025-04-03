using Aggregations
using Distances
using DifferentialEquations
using NonlinearSolve, SteadyStateDiffEq, OrdinaryDiffEq
using LinearAlgebra
using PlotlyJS
using Statistics
#


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
    list = zeros(length(vec_point[:,1]))
    i = 1
    for point2 in eachrow(vec_point)
        list[i] = norm(point1 - point2)
        i = i +1
    end
    return list
    
end

#find the 3 closest points for every steady state
function find_k_closest_point(points,k)
    matrix = zeros(Int,length(points[:,1]),k)
    
    for i in range(1,length(points[:,1]))

        distances = sortperm(compute_all_distance(points[i,:], points))
        #matrix[i,1] = i 
        for j in range(1,k)
            matrix[i,j] = distances[j + 1]
        end
    end
    return matrix
end

function generate_randoms()
    list = []
    for i in range(1,20)
        u0 = u0 = vec(2 * rand(3, N) .- 1)
        append!(list,u0)
        #println(u0)
    end
end

function solver(α,λ,u)
    p = (α, λ, N, r, f, a, XX, YY, ZZ, YX, ZX, ZY, Xa, Ya)
    u = solve(SteadyStateProblem(ff_hessian, u[:], p), DynamicSS(TRBDF2()))
    u = reshape(u,N,3)
    return u
end

function plot_steadystates(u,α,λ)
    center_of_mass = [mean(u[:,1]), mean(u[:,2]), mean(u[:,3])]

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
                    size=6,
                    color="blue", 
                    opacity=0.3
                ),
                name="Steady States",
                type = "scatter3d"
            )

            #lines between points
            
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

function sorted_eigenvalues(h,u,p)
    return sort((eigen(h(u,p)).values))
end


function add_distance(arr,u,type)
    center_of_mass = [mean(u[:,1]), mean(u[:,2]), mean(u[:,3])]
    if type == "max"
        max_dist = compute_maximum_distance(center_of_mass,u)
        push!(arr,max_dist)
    
    elseif type == "min"
        min_dist = compute_minimum_distance(center_of_mass,u)
        push!(arr,min_dist)
    
    elseif type == "diff"
        max_dist = compute_maximum_distance(center_of_mass,u)
        min_dist = compute_minimum_distance(center_of_mass,u)
        push!(arr,max_dist - min_dist)
    end

    return arr

end

function plot_distances(α_start, α_finish, step, λ, max_arr, min_arr)
    x_values = range(α_start, stop = α_finish, step = step)
    y_min = min_arr
    y_max = max_arr

    trace1 = scatter(x=x_values, y= y_min,
                    mode="lines+markers",
                    name="line_min")
    trace2 = scatter(x=x_values, y=y_max,
                    mode="lines+markers",
                    name="line_max")
    
    line_plot = plot([trace1,trace2])
    display(line_plot)
end

function simulations(α_start, α_finish, step,u0)
    
    num_steps = Int(div(α_finish - α_start, step) + 2 ) # Add 1 to include the last step
    min_arr = []
    max_arr = []

    #main loop for running simulations
    for α in α_start : step : α_finish
        u = solver(α,λ,u0[:])
        min_arr = add_distance(min_arr,u,"min")
        max_arr = add_distance(max_arr,u,"max")
        #plot_steadystates(u,α,λ)
        
        
        u0 = u
    end
    println(min_arr)
    println(max_arr)
    plot_distances(α_start, α_finish, step, λ, max_arr, min_arr)
end








    
        






u0 = vec(2 * rand(3, N) .- 1)
simulations(25,30,5,u0)
#u = solver(30,0.1,u0)
#plot_steadystates(u)


