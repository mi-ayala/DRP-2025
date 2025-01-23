#### These are the implementions of the energy, gradient and hessian.


function e(u, p)

    α, λ, N = p


    r = zeros(eltype(u[1]), N, N)
    ener = zero(eltype(u[1]))

    ### Computing the distances and forces
    pairwise!(r, Euclidean(), transpose(reshape(u, :, 3)), dims=2)


    @inbounds for i in 1:(N-1)

        for j in (i+1):N

            ener += (1 / α) * (r[i, j]^α) + (1 / λ) * (r[i, j]^(-λ))

        end

    end

    return ener / N
end


function g(u, α, λ, N)

    u = transpose(reshape(u, :, 3))
    r = zeros(eltype(α), N, N)
    du = zeros(eltype(α), 3, N)

    pairwise!(r, Euclidean(), u, dims=2)

    ### The main loop
    @inbounds for i in 1:(N-1)

        @inbounds for j in (i+1):N

            Interaction = (-r[i, j]^(α - 2) + r[i, j]^(-λ - 2)) * (view(u, :, i) - view(u, :, j))

            du[:, i] += Interaction

            du[:, j] -= Interaction

        end


    end

    reshape((1 / N) .* transpose(du), :, 1)

end

function g(u, p)

    α, λ, N = p


    u = transpose(reshape(u, :, 3))

    r = zeros(eltype(u[1]), N, N)
    du = zeros(eltype(u[1]), 3, N)

    pairwise!(r, Euclidean(), u, dims=2)

    ### The main loop
    @inbounds for i in 1:(N-1)

        @inbounds for j in (i+1):N

            Interaction = (-r[i, j]^(α - 2) + r[i, j]^(-λ - 2)) * (view(u, :, i) - view(u, :, j))

            du[:, i] += Interaction

            du[:, j] -= Interaction

        end


    end

    reshape((1 / N) .* transpose(du), :, 1)

end


function h(u, p)


    α, λ, N = p

    type = eltype(u[1])

    u = transpose(reshape(u, :, 3))
    r = zeros(type, N, N)
    pairwise!(r, Euclidean(), u, dims=2)

    f = -r .^ (α - 2) + r .^ (-λ - 2)
    a = -(α - 2) .* r .^ (α - 4) + (-λ - 2) .* r .^ (-λ - 4)

    @inbounds for i in 1:N

        f[i, i] = zero(type)
        a[i, i] = zero(type)

    end

    X_x = zeros(type, N, N)
    Y_y = zeros(type, N, N)
    X_y = zeros(type, N, N)
    X_z = zeros(type, N, N)
    Y_z = zeros(type, N, N)
    Z_z = zeros(type, N, N)

    X = @views repeat(u[1, :], 1, N) - repeat(transpose(u[1, :]), N, 1)
    Y = @views repeat(u[2, :], 1, N) - repeat(transpose(u[2, :]), N, 1)
    Z = @views repeat(u[3, :], 1, N) - repeat(transpose(u[3, :]), N, 1)


    @inbounds for i in 1:N


        J = zeros(type, N, N)
        J[i, :] .= one(type)
        J[:, i] .= -one(type)

        X_x[:, i] .= sum(J .* (X .* X .* a + f), dims=2)
        Y_y[:, i] .= sum(J .* (Y .* Y .* a + f), dims=2)
        Z_z[:, i] .= sum(J .* (Z .* Z .* a + f), dims=2)

        X_y[:, i] .= sum(J .* (Y .* X .* a), dims=2)
        X_z[:, i] .= sum(J .* (Z .* X .* a), dims=2)
        Y_z[:, i] .= sum(J .* (Z .* Y .* a), dims=2)


    end

    return [X_x X_y X_z; X_y Y_y Y_z; X_z Y_z Z_z] / N

end


function g!(du, u, p, t)

    α, λ, N = p

    u = transpose(reshape(u, :, 3))
    r = zeros(eltype(u[1]), N, N)

    du_reshaped = transpose(reshape(du, :, 3))
    pairwise!(r, Euclidean(), u, dims=2)

    ### The main loop
    @inbounds for i in 1:(N-1)

        @inbounds for j in (i+1):N

            Interaction = (-r[i, j]^(α - 2) + r[i, j]^(-λ - 2)) * (view(u, :, i) - view(u, :, j))

            du_reshaped[:, i] += Interaction

            du_reshaped[:, j] -= Interaction

        end


    end

    mul!(du, du, 1 / N)

    nothing
end


function h!(Jac, u, p, t)


    α, λ, N = p

    type = eltype(u[1])

    u = transpose(reshape(u, :, 3))
    r = zeros(type, N, N)
    pairwise!(r, Euclidean(), u, dims=2)

    f = -r .^ (α - 2) + r .^ (-λ - 2)
    a = -(α - 2) .* r .^ (α - 4) + (-λ - 2) .* r .^ (-λ - 4)

    @inbounds for i in 1:N

        f[i, i] = zero(type)
        a[i, i] = zero(type)

    end

    X_x = view(Jac, 1:N, 1:N)
    X_y = view(Jac, 1:N, N+1:2N)
    X_z = view(Jac, 1:N, 2N+1:3N)

    Y_x = view(Jac, N+1:2N, 1:N)
    Y_y = view(Jac, N+1:2N, N+1:2N)
    Y_z = view(Jac, N+1:2N, 2N+1:3N)

    Z_x = view(Jac, 2N+1:3N, 1:N)
    Z_y = view(Jac, 2N+1:3N, N+1:2N)
    Z_z = view(Jac, 2N+1:3N, 2N+1:3N)


    X = @views repeat(u[1, :], 1, N) - repeat(transpose(u[1, :]), N, 1)
    Y = @views repeat(u[2, :], 1, N) - repeat(transpose(u[2, :]), N, 1)
    Z = @views repeat(u[3, :], 1, N) - repeat(transpose(u[3, :]), N, 1)


    J = zeros(type, N, N)

    @inbounds for i in 1:N

        J .= zero(type)

        J[i, :] .= one(type)
        J[:, i] .= -one(type)

        X_x[:, i] .= sum(J .* (X .* X .* a + f), dims=2)
        Y_y[:, i] .= sum(J .* (Y .* Y .* a + f), dims=2)
        Z_z[:, i] .= sum(J .* (Z .* Z .* a + f), dims=2)

        X_y[:, i] .= sum(J .* (Y .* X .* a), dims=2)
        X_z[:, i] .= sum(J .* (Z .* X .* a), dims=2)
        Y_z[:, i] .= sum(J .* (Z .* Y .* a), dims=2)


    end

    Y_x .= X_y
    Z_x .= X_z
    Z_y .= Y_z


    mul!(Jac, Jac, 1 / N)

    nothing
end



function h_fast!(Jac, u, p, t)


    ### Caches f,a,J

    α, λ, N, r, f, a, XX, YY, ZZ, YX, ZX, ZY, Xa, Ya = p

    type = eltype(u[1])
    u = transpose(reshape(u, :, 3))
    pairwise!(r, Euclidean(), u, dims=2)


    ### First the colums because Julia is column major
    f .= zero(type)
    a .= zero(type)


    @inbounds for j in 2:N

        @inbounds for i in 1:j-1
            f[i, j] = -r[i, j]^(α - 2) + r[i, j]^(-λ - 2)
            f[j, i] = f[i, j]

            a[i, j] = -(α - 2) * r[i, j]^(α - 4) + (-λ - 2) * r[i, j]^(-λ - 4)
            a[j, i] = a[i, j]

        end

    end

    Jac .= zero(type)

    X_x = view(Jac, 1:N, 1:N)
    X_y = view(Jac, 1:N, N+1:2N)
    X_z = view(Jac, 1:N, 2N+1:3N)

    Y_x = view(Jac, N+1:2N, 1:N)
    Y_y = view(Jac, N+1:2N, N+1:2N)
    Y_z = view(Jac, N+1:2N, 2N+1:3N)

    Z_x = view(Jac, 2N+1:3N, 1:N)
    Z_y = view(Jac, 2N+1:3N, N+1:2N)
    Z_z = view(Jac, 2N+1:3N, 2N+1:3N)


    X = @views repeat(u[1, :], 1, N) - repeat(transpose(u[1, :]), N, 1)
    Y = @views repeat(u[2, :], 1, N) - repeat(transpose(u[2, :]), N, 1)
    Z = @views repeat(u[3, :], 1, N) - repeat(transpose(u[3, :]), N, 1)

    Xa = X .* a
    Ya = Y .* a

    ### This we could threaded
    XX = X .* Xa + f
    YY = Y .* Ya + f
    ZZ = Z .* Z .* a + f
    YX = Y .* Xa
    ZX = Z .* Xa
    ZY = Z .* Ya

    ### To get rid of the J we need to do the diagonal in a different loop I think

    @inbounds for i in 1:N

        X_x[i, i] = sum(view(XX, i, :))
        X_x[:, i] -= view(XX, :, i)

        Y_y[i, i] = sum(view(YY, i, :))
        Y_y[:, i] -= view(YY, :, i)

        Z_z[i, i] = sum(view(ZZ, i, :))
        Z_z[:, i] -= view(ZZ, :, i)

        X_y[i, i] = sum(view(YX, i, :))
        X_y[:, i] -= view(YX, :, i)

        X_z[i, i] = sum(view(ZX, i, :))
        X_z[:, i] -= view(ZX, :, i)

        Y_z[i, i] = sum(view(ZY, i, :))
        Y_z[:, i] -= view(ZY, :, i)

    end

    Y_x .= X_y
    Z_x .= X_z
    Z_y .= Y_z


    mul!(Jac, Jac, 1 / N)

    nothing
end


