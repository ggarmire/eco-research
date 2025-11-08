#4D competitive Lotka–Volterra (Vano 2006)
# Rainer Engelken (2025)

using Random
using LinearAlgebra
using PyPlot

"""
    lv4d(Tsim::Float64, dt::Float64, ONSstep::Int, seed::Int)

4D competitive Lotka–Volterra (Vano 2006)
Returns (ls_avg::Vector{Float64}, X::Array{Float64,2}, lle::Vector{Float64}).

- `ls_avg`: time-averaged Lyapunov spectrum (continuous-time units)
- `X`: trajectory, size (nstep, 4)
- `lle`: sequence of (1/dt) * log|R[1,1]| recorded every `ONSstep`
"""
function lv4d(Tsim::Float64, dt::Float64, ONSstep::Int, seed::Int)


    # Parameters from Vano et al. (2006), Eq. (3)
    r = [1.0, 0.72, 1.53, 1.27]
    A = [
        1.00   1.09   1.52   0.00;
        0.00   1.00   0.44   1.36;
        2.33   0.00   1.00   0.47;
        1.21   0.51   0.35   1.00
    ]

    nstep = ceil(Int, Tsim / dt)
    Random.seed!(seed)

    # Initial condition: x = 0.5 + 0.1 * rand
    x = 0.5 .+ 0.1 .* rand(4)

    # Random orthonormal Q via QR of a random 4x4
    Q0 = randn(4,4)
    qrf0 = qr(Q0)
    Q = Matrix(qrf0.Q)

    X = Array{Float64}(undef, nstep, 4)
    lle = Float64[]
    ls = zeros(4)
    I4 = Matrix{Float64}(I, 4, 4)

    Ax = similar(x)
    f  = similar(x)
    J  = zeros(4,4)
    D  = similar(J)
   @time for n in 1:nstep
        # Vector field: dx/dt = r .* x .* (1 .- A*x)
        Ax .= A * x
        @inbounds @simd for i in 1:4
            f[i] = r[i] * x[i] * (1.0 - Ax[i])
        end

        # Euler step
        x .= x .+ dt .* f

        # Clamp to non-negative domain
        @inbounds @simd for i in 1:4
            if x[i] < 0.0
                x[i] = 0.0
            end
        end

        #X[n, :] .= x

        # Jacobian: J[i,j] = r[i] * ( δ_ij*(1 - (A*x)[i]) - x[i]*A[i,j] )
        one_minus_Ax_i = 0.0
        @inbounds for i in 1:4
            one_minus_Ax_i = 1.0 - Ax[i]
            @inbounds @simd for j in 1:4
                J[i,j] = r[i] * ( (i == j ? one_minus_Ax_i : 0.0) - x[i]*A[i,j] )
            end
        end

        # Discrete tangent map for Euler step
        D .= I4 .+ dt .* J
        Q .= D * Q

        if (n % ONSstep) == 0
            qrf = qr(Q)
            Q   = Matrix(qrf.Q)
            R   = qrf.R
            diagR = abs.(diag(R))
            ls .+= log.(diagR)
            push!(lle, (1.0 / dt) * log(abs(R[1,1])))
        end
    end

    ls_avg = ls ./ (dt * nstep)
   # @show ls_avg
    return ls_avg[1]#, X, lle
end

tRange=10.0 .^(1.5:0.25:4);dtRange=10.0 .^(-3:-3);ONSstep=1000;seed=collect(1:3);lsMax = lv4d.(tRange', dtRange, ONSstep, seed); figure();semilogx(tRange,lsMax')

