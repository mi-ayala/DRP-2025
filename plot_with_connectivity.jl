using GLMakie


function plot_steady_state(x, connectivity)

    if size(x, 2) == 1

        x = reshape(x, :, 3)

    end


    function basic_plot(x, connectivity)

        NumParticles = convert(Int, length(x[:, 1]))

        fig = Figure(size=(1200, 1200), fontsize=22)
        ax = Axis3(fig[1, 1], aspect=:data)
        hidedecorations!(ax)
        hideydecorations!(ax, ticks=false)
        colors = ["#FF410DFF", "#6EE2FFFF", "#F7C530FF", "#95CC5EFF",
            "#D0DFE6FF", "#F79D1EFF", "#748AA6FF"]

        for j = 1:NumParticles
            for k = 1:3
                linesegments!(ax, [x[j, 1], x[connectivity[j, k], 1]], [x[j, 2], x[connectivity[j, k], 2]], [x[j, 3], x[connectivity[j, k], 3]], color=colors[1], linewidth=3)
            end
        end

        meshscatter!(x[:, 1], x[:, 2], x[:, 3], color=colors[1], markersize=0.1)

        fig

    end


    with_theme(theme_black()) do

        basic_plot(x, connectivity)

    end


end