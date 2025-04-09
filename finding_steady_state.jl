using Aggregations
using Distances
using DifferentialEquations
using NonlinearSolve, SteadyStateDiffEq, OrdinaryDiffEq
using LinearAlgebra
using PlotlyJS
using Statistics
using JLD2
#


include("functions.jl")

### Parameters 
N = 60

### Kernel Parameters
λ = 0.01
α = 2



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
        if current_distance < min_distance && point1 != point2
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

function generate_randoms(N,k)
    list = zeros(3,N,k)
    for i in range(1,k)
        u0 = vec(2 * rand(3, N) .- 1)
        list[:,:,i] = u0
        #println(u0)
    end
    return list
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

    elseif type == "mean"
        Mean = mean(compute_all_distance(center_of_mass,u))
        push!(arr,Mean)
    elseif type == "std"
        Std = std(compute_all_distance(center_of_mass,u))
        push!(arr,Std)

    end

    return arr

end

function trace_measurement(α_start, α_finish, step, arr,name)
    

    x_values = range(α_start, stop = α_finish, step = step)
    y_values = arr

    trace = scatter(x=x_values, y= y_values,
                    mode="lines+markers",
                    name="$name distance")
    return trace
end

function compute_all_minimum_relative_distance(u)
    list = []
    for point in eachrow(u)
        push!(list,compute_minimum_distance(point,u))
    end
    return list


end

function trace_minimum_relative_distance(α,arr)
    hist = histogram(x=arr,marker = attr(
                                    color = "skyblue",  # fill color
                                    line = attr(color = "black", width = 1.5)  # border color + thickness
                                    ),)
    return hist
end

function neighbors(point, u, r)
    count = 0
    for p in eachrow(u)
        d = norm(point - p)
        if 0 < d < r   # Exclude the point itself
            count += 1
        end
    end
    return count
end

function compute_num_neighbors(u,r)
    list = []
    for point in eachrow(u)
        push!(list,neighbors(point,u,r))
    end
    return list
end


function compute_energy(k)
    initial_conditions = generate_randoms(N,k)
    min_u = vec(2 * rand(3, N) .- 1)
    min_energy = 10000000
    for i in range(1,k)
        u = solver(2,0.01,initial_conditions[:,:,i])
        energy = e(u,p)
        if energy < min_energy
            min_energy = energy
            min_u = u
        end
        print("energy: $energy, norm : ")
        println((norm(g(u,p), Inf)))
    end
    return min_u

end

function simulations(α_start, α_finish, step,u0)
    
    num_steps = Int(div(α_finish - α_start, step) + 2 ) # Add 1 to include the last step
    min_arr = []
    max_arr = []
    mean_arr = []
    std_arr = []

    #main loop for running simulations
    for α in α_start : step : α_finish
        u = solver(α,λ,u0[:])
        min_arr = add_distance(min_arr,u,"min")
        max_arr = add_distance(max_arr,u,"max")
        mean_arr = add_distance(mean_arr,u,"mean")
        std_arr = add_distance(std_arr,u,"std")
        u0 = u
        trace_min_rel = trace_minimum_relative_distance(α,compute_all_minimum_relative_distance(u0))
        layout = Layout(
            title = "Distribution of minimum relative distances for all steady states at α = $α",
            xaxis_title = "Count",
            yaxis_title = "Minimum relative distance"
        )
        
        display(plot(trace_min_rel,layout))
    end

    
    

    trace_max = trace_measurement(α_start, α_finish,step,max_arr,"max")
    trace_min = trace_measurement(α_start, α_finish,step,min_arr,"min")
    trace_mean = trace_measurement(α_start, α_finish,step,mean_arr,"mean")
    #trace_std = trace_measurement(α_start, α_finish,step,std_arr,"std")
    layout = Layout(
            title = "Measurements of steady states distances over the variation of α",
            xaxis_title = "Value of α",
            yaxis_title = "Distance",
            showlegend = true
            )
    p = plot([trace_max,trace_min,trace_mean],layout)
    display(p)

    
    
    
    
    
    
end








    
        




#min_initial_condition = compute_energy(500)
#save("minimum_initial_condition.jld2", "minimum_initial_condition", min_initial_condition)
data = load("minimum_initial_condition.jld2")
min_initial_condition = data["minimum_initial_condition"]
#println("finished")
#println(min_initial_condition)
#println(compute_energy())
#u0 = vec(2 * rand(3, N) .- 1)
#point = u0[1,:]
#neighbors(point,u0,0.025)
#arr = compute_all_minimum_relative_distance(u0)
#plot(trace_measurement(1,2,2,arr,"relative_min"))

simulations(25,30,5,min_initial_condition)
#u = solver(30,0.1,u0)
#plot_steadystates(u)


