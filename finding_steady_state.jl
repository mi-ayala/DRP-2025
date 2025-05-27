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
N = 50

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

#find the k closest points for every steady state
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

#generate k random initial conditions
function generate_randoms(N,k)
    list = zeros(3,N,k)
    for i in range(1,k)
        u0 = vec(2 * rand(3, N) .- 1)
        list[:,:,i] = u0
        #println(u0)
    end
    return list
end

#solves the ODE system
function solver(α,λ,u)
    p = (α, λ, N, r, f, a, XX, YY, ZZ, YX, ZX, ZY, Xa, Ya)
    u = solve(SteadyStateProblem(ff_hessian, u[:], p), DynamicSS(TRBDF2()))
    u = reshape(u,N,3)
    #println("norm when α = $α: $(norm(g(u,p), Inf))")
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
                    opacity=0.6
                ),
                name="Steady States",
                type = "scatter3d",
                
            )

            #lines between points
            
        ], 
        Layout(
            title=attr(
             text=" Position of All Steady States when α = $α, λ = $λ",   
             x=0.47,                    # Position title to the far right
             y=0.92,                    # Position title near the top
             #xanchor="middle",           # Right align the title
             #yanchor="top"              # Top align the title
            ),
            legend=attr(
                x=0.67,           # 0 = left, 1 = right
                y=0.5,           # 0 = bottom, 1 = top
            
            ),
            font=attr(size = 18),
           # bgcolor="rgba(255,255,255,0.9)",  # Light background with some opacity
            #bordercolor="black",
            #borderwidth=2,

            margin=attr(l=0, r=0, b=50, t=0),
            scene = attr(
                xaxis = attr(showbackground=false, showgrid=true, zeroline=false, showticklabels=false, title=""),
                yaxis = attr(showbackground=false, showgrid=true, zeroline=false, showticklabels=false,title=""),
                zaxis = attr(showbackground=false, showgrid=true, zeroline=false, showticklabels=false,title=""),
                bgcolor = "rgba(0,0,0,0)"  # Transparent background
                ),
                showlegend=true,
                
        ))
            
        
       display(p)
end

function sorted_eigenvalues(h,u,p)
    return sort((eigen(h(u,p)).values))
end

#add the steady states distances to an array
function add_distance(arr,u,type)
    center_of_mass = [mean(u[:,1]), mean(u[:,2]), mean(u[:,3])]
    if type == "Max"
        max_dist = compute_maximum_distance(center_of_mass,u)
        push!(arr,max_dist)
    
    elseif type == "Min"
        min_dist = compute_minimum_distance(center_of_mass,u)
        push!(arr,min_dist)
    
    elseif type == "diff"
        max_dist = compute_maximum_distance(center_of_mass,u)
        min_dist = compute_minimum_distance(center_of_mass,u)
        push!(arr,max_dist - min_dist)

    elseif type == "Mean"
        Mean = mean(compute_all_distance(center_of_mass,u))
        push!(arr,Mean)
    elseif type == "Std"
        Std = std(compute_all_distance(center_of_mass,u))
        push!(arr,Std)

    end

    return arr

end

#creates the trace for plotting the different distances
function trace_measurement(α_start, α_finish, step, arr,name)
    

    x_values = range(α_start, stop = α_finish, step = step)
    y_values = arr

    trace = scatter(x=x_values, y= y_values,
                    mode="lines+markers",
                    name="$name Distance")
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

#compute how many other points are in a neighborhood of radius r
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


#main loop for the simulations
function simulations(α_start, α_finish, step,u0)
    
    num_steps = Int(div(α_finish - α_start, step) + 2 ) # Add 1 to include the last step
    min_arr = []
    max_arr = []
    mean_arr = []
    std_arr = []


    min_arr2 = []
    max_arr2 = []
    mean_arr2 = []
    std_arr2 = []
    

    #main loop for running simulations
    for α in α_start : step : α_finish
        u = solver(α,λ,u0[:])
        min_arr = add_distance(min_arr,u,"Min")
        max_arr = add_distance(max_arr,u,"Max")
        mean_arr = add_distance(mean_arr,u,"Mean")
        std_arr = add_distance(std_arr,u,"Std")

       
        u0 = u


            widths = compute_widths(500,u0)
            push!(min_arr2,minimum(widths))
            push!(max_arr2,maximum(widths))
            push!(mean_arr2,mean(widths))
            push!(std_arr2,std(widths))

            
        if α == 30 || α == 70 || α == 200 || α == 300 || α == 500 || α == 1000 || α == 2000

        #plot_steadystates(u0,α,λ)
            plot_histogram_width(u0,α)
        end
        
    end
    


    
    

    trace_max = trace_measurement(α_start, α_finish,step,max_arr,"Max")
    trace_min = trace_measurement(α_start, α_finish,step,min_arr,"Min")
    trace_mean = trace_measurement(α_start, α_finish,step,mean_arr,"Mean")
    trace_std = trace_measurement(α_start, α_finish,step,std_arr,"Std")
    layout = Layout(
            title = "Measurements of Steady States Distances over the Variation of α (N = $N)",
            xaxis_title = "Value of α",
            yaxis_title = "Distance",
            showlegend = true
            )
    p = plot([trace_max,trace_min,trace_mean,trace_std],layout)
    display(p)

    trace_max2 = trace_measurement(α_start, α_finish,step,max_arr2,"Max")
    trace_min2 = trace_measurement(α_start, α_finish,step,min_arr2,"Min")
    trace_mean2 = trace_measurement(α_start, α_finish,step,mean_arr2,"Mean")
    trace_std2 = trace_measurement(α_start, α_finish,step,std_arr2,"Std")

    layout2 = Layout(
            title = "Measurements of Widths Distances over the Variation of α (N = $N)",
            xaxis_title = "Value of α",
            yaxis_title = "Distance",
            showlegend = true
            )
    p2 = plot([trace_max2,trace_min2,trace_mean2,trace_std2],layout2)
    display(p2)

end

function compute_width(d,points)
    projections = []
    for point in eachrow(points)
        push!(projections,dot(point,d))
    end
    return maximum(projections) - minimum(projections)
end

function compute_widths(num_samples, points)
    widths = []
    for i in 1:num_samples
        d = rand(1,3)
        d /= norm(d)
        push!(widths,compute_width(d,points)) 
    end
    return widths
end

function plot_histogram_width(u0,α)
    widths = compute_widths(500,u0)
            println("std of width for α = $α:", std(widths))
            println("mean of width for α = $α:", mean(widths))
            println("median of width for α = $α:", median(widths))
    
            layout = Layout(
                title = "Distribution of Widths for a Random Sample of 500 Unit Directions (α = $α)",
                 xaxis_title = "Width",
                 yaxis_title = "Count",
                bargap = 0.05,
            )

            trace_hist = histogram(x = widths,  marker = attr(
                        color = "skyblue",  # fill color
                        line = attr(
                        color = "black", # outline color
                        width = 1        # outline thickness
                            )
                         )
                    )
    display(plot(trace_hist,layout))
    
end

#u0 = rand(N,3)
#d = [0,0,1]
#println(compute_width(d,u0))










    
        




#min_initial_condition = compute_energy(500)
#save("minimum_initial_condition.jld2", "minimum_initial_condition", min_initial_condition)
data = load("minimum_initial_condition.jld2")
min_initial_condition = data["minimum_initial_condition"]
#print(compute_num_neighbors(min_initial_condition,0.5))
#println("finished")
#println(min_initial_condition)
#println(compute_energy())
u0 = vec(2 * rand(3, N) .- 1)
#point = u0[1,:]
#neighbors(point,u0,0.025)
#arr = compute_all_minimum_relative_distance(u0)
#plot(trace_measurement(1,2,2,arr,"relative_min"))

#simulations(20,20,5,min_initial_condition)
simulations(20,70,5,u0)
#u = solver(30,0.1,u0)
#plot_steadystates(u)


