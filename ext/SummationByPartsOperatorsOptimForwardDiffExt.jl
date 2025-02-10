module SummationByPartsOperatorsOptimForwardDiffExt

using Optim: Optim, Options, LBFGS, optimize, minimizer
import ForwardDiff

using SummationByPartsOperators: SummationByPartsOperators, GlaubitzNordströmÖffner2023, GlaubitzIskeLampertÖffner2024,
                                 MatrixDerivativeOperator, MultidimensionalMatrixOperator
using SummationByPartsOperators: get_nsigma # TODO: Only temporary
using LinearAlgebra: Diagonal, LowerTriangular, dot, diag, norm, mul!
using SparseArrays: spzeros
using PreallocationTools: DiffCache, get_tmp

function vandermonde_matrix(functions, nodes)
    N = length(nodes)
    K = length(functions)
    T = typeof(functions[1](nodes[1]))
    V = zeros(T, N, K)
    for i in 1:N
        for j in 1:K
            V[i, j] = functions[j](nodes[i])
        end
    end
    return V
end

function create_S(sigma, N, bandwidth, size_boundary, different_values)
    S = zeros(eltype(sigma), N, N)
    set_S!(S, sigma, N, bandwidth, size_boundary, different_values)
    return S
end

# M has to be square
function set_skew_symmetric!(M, sigma, init_k = 1)
    N = size(M, 1)
    k = init_k
    for i in 1:N
        for j in (i + 1):N
            M[i, j] = sigma[k]
            M[j, i] = -sigma[k]
            k += 1
        end
    end
    return k
end

# D has to be square
function set_banded!(D, sigma, bandwidth, init_k = 1, different_values = true)
    N = size(D, 1)
    k = init_k
    for i in 1:N
        for j in (i + 1):N
            if j - i <= bandwidth
                if different_values
                    l = k
                    k += 1
                else
                    l = init_k + j - i - 1
                end
                D[i, j] = sigma[l]
                D[j, i] = -sigma[l]
            end
        end
    end
    return k
end

function set_triangular!(C, sigma, bandwidth, size_boundary = 2 * bandwidth, init_k = 1, different_values = true)
    N = size(C, 1)
    k = init_k
    if different_values
        start_i = N - bandwidth + 1
    else
        start_i = size_boundary - bandwidth + 1
    end
    for i in start_i:N
        for j in 1:(i - start_i + 1)
            if different_values
                l = k
            else
                l = init_k - 1 + bandwidth + j - (i - start_i + 1)
            end
            C[i, j] = sigma[l]
            k += 1
        end
    end
    return k
end

permute_rows_and_cols(P) = P[size(P, 1):-1:1, size(P, 2):-1:1]

@views function set_S!(S, sigma, N, bandwidth, size_boundary = 2 * bandwidth, different_values = true)
    if bandwidth == N - 1
        set_skew_symmetric!(S, sigma)
    else
        b = bandwidth
        c = size_boundary
        # upper left boundary block
        M1 = S[1:c, 1:c]
        k = set_skew_symmetric!(M1, sigma, 1)
        # lower right boundary block
        M2 = S[(N - c + 1):N, (N - c + 1):N]
        if different_values
            k = set_skew_symmetric!(M2, sigma, k)
        else
            M2 .= -permute_rows_and_cols(M1)
        end

        # banded matrix in the middle
        D = S[(c + 1):(N - c), (c + 1):(N - c)]
        k = set_banded!(D, sigma, b, k, different_values)

        # upper central block with triangular part
        C1 = S[1:c, (c + 1):(N - c)]
        k = set_triangular!(C1, sigma, b, c, k, different_values)
        # central left block with triangular part
        S[(c + 1):(N - c), 1:c] = -C1'
        # central right block with triangular part
        C2 = S[(c + 1):(N - c), (N - c + 1):N]
        if different_values
            k = set_triangular!(C2, sigma, b, c, k, different_values)
            # lower central block with triangular part
            S[(N - c + 1):N, (c + 1):(N - c)] = -C2'
        else
            C1_bar = permute_rows_and_cols(C1)
            C2 .= C1_bar'
            S[(N - c + 1):N, (c + 1):(N - c)] = -C1_bar
        end
    end
end

sig(x) = 1 / (1 + exp(-x))
sig_deriv(x) = sig(x) * (1 - sig(x))
invsig(p) = log(p / (1 - p))

# leading to softmax
# sig(x) = exp(x)
# sig_deriv(x) = exp(x)
# invsig(p) = log(p)

function create_P(rho, vol)
    P = Diagonal(sig.(rho))
    P *= vol / sum(P)
    return P
end

function create_B(phi, normals, on_boundary, i)
    b = zeros(eltype(phi), length(on_boundary))
    B = Diagonal(b)
    set_B!(B, phi, normals, on_boundary, i)
    return B
end

function set_B!(B, phi, normals, on_boundary, i)
    j = 1
    for k in eachindex(on_boundary)
        if on_boundary[k]
            B[k, k] = phi[j] * normals[j][i]
            j += 1
        else
            B[k, k] = zero(eltype(phi))
        end
    end
end


function SummationByPartsOperators.get_nsigma(N, bandwidth, size_boundary = 2 * bandwidth, different_values = true)
    if bandwidth == N - 1
        # whole upper right triangle
        return div(N * (N - 1), 2)
    else
        if different_values
            # upper right corner for boundary blocks cxc each: c*(c - 1)/2
            # lower triangle including diagonal for two different upper and right central blocks bxb each: b*(b + 1)/2
            # non-repeating stencil for diagonal block: (N - 2c - b)b + b*(b - 1)/2 = Nb - 1/2(4c*b + b^2 + b)
            # => in total: Nb + 1/2b^2 + c^2 - 2c*b - c + 1/2b
            # return N * bandwidth + div(bandwidth * (bandwidth - 3), 2) # for c = 2b
            b = bandwidth
            c = size_boundary
            return N * b + div(b * (b + 1), 2) + c^2 - 2 * b * c - c
        else
            # upper right corner for boundary blocks cxc: c*(c - 1)/2 plus b from repeating stencil
            # => in total: c*(c - 1)/2 + b
            # return 2 * bandwidth^2 # for c = 2b
            return div(size_boundary * (size_boundary - 1), 2) + bandwidth
        end
    end
end

function SummationByPartsOperators.multidimensional_function_space_operator(basis_functions, nodes, on_boundary, normals, moments, vol,
                                                                            source::SourceOfCoefficients;
                                                                            derivative_order = 1, accuracy_order = 0, bandwidth = length(nodes) - 1,
                                                                            size_boundary = 2 * bandwidth, different_values = true,
                                                                            opt_alg = LBFGS(), options = Options(g_tol = 1e-14, iterations = 10000),
                                                                            x0 = nothing, verbose = false) where {SourceOfCoefficients}

    if derivative_order != 1
        throw(ArgumentError("Derivative order $derivative_order not implemented."))
    end
    if (length(nodes) < 2 * size_boundary + bandwidth || bandwidth < 1) && (bandwidth != length(nodes) - 1)
        throw(ArgumentError("2 * size_boundary + bandwidth = $(2 * size_boundary + bandwidth) needs to be smaller than or equal to N = $(length(nodes)) and bandwidth = $bandwidth needs to be at least 1."))
    end
    weights, weights_boundary, Ds = construct_multidimensional_function_space_operator(basis_functions, nodes, on_boundary, normals, moments, vol, source;
                                                                                       bandwidth, size_boundary, different_values,
                                                                                       opt_alg, options, x0, verbose)
    return MultidimensionalMatrixOperator(nodes, on_boundary, normals, weights, weights_boundary, Ds, accuracy_order, source)
end

function construct_multidimensional_function_space_operator(basis_functions, nodes, on_boundary, normals, moments, vol,
                                                            ::GlaubitzIskeLampertÖffner2024;
                                                            bandwidth = length(nodes) - 1,
                                                            size_boundary = 2 * bandwidth, different_values = true,
                                                            opt_alg = LBFGS(), options = Options(g_tol = 1e-14, iterations = 10000),
                                                            x0 = nothing, verbose = false)
    T = typeof(basis_functions[1](nodes[1]))
    d = length(first(nodes))
    K = length(basis_functions)
    N = length(nodes)
    N_boundary = sum(on_boundary)
    @assert length(normals) == N_boundary "You must provide normals for all boundary nodes (length(normals) = $(length(normals)), N_boundary = $N_boundary)."
    L = get_nsigma(N, bandwidth, size_boundary, different_values)
    basis_functions_gradients = [x -> ForwardDiff.gradient(basis_functions[i], x) for i in 1:K]
    # TODO: Orthonormalize? What happens with moments? Need moments with respect to orthonormalized basis functions?
    V = vandermonde_matrix(basis_functions, nodes)
    V_xis = ntuple(j -> vandermonde_matrix([x -> basis_functions_gradients[i](x)[j] for i in 1:K], nodes), d)

    S = zeros(T, N, N)
    A = zeros(T, N, K)
    M = zeros(T, K, K)
    S_cache = DiffCache(S)
    A_cache = DiffCache(A)
    SV_cache = DiffCache(copy(A))
    PV_xi_cache = DiffCache(copy(A))
    B_cache = DiffCache(copy(S))
    BV_cache = DiffCache(copy(A))
    VTBV_cache = DiffCache(M)
    C_cache = DiffCache(copy(M))
    p = (V, V_xis, normals, moments, on_boundary, vol, S_cache, A_cache, SV_cache, PV_xi_cache, B_cache, BV_cache, C_cache, VTBV_cache, bandwidth, size_boundary, different_values)
    if isnothing(x0)
        # x0 = zeros(T, d * L + N + N_boundary)
        x0 = [zeros(T, d * L); invsig.(1/N * ones(T, N)); zeros(T, N_boundary)]
    else
        n_total = d * L + N + N_boundary
        @assert length(x0) == n_total "Initial guess has be d * L + N + N_boundary = $n_total long, but got length $(length(x0))"
    end

    f(x) = SummationByPartsOperators.multidimensional_optimization_function(x, p)
    result = optimize(f, x0, opt_alg, options; autodiff = :forward)
    verbose && show(stdout, "text/plain", result)

    x = minimizer(result)
    rho = x[(end - N - N_boundary + 1):(end - N_boundary)]
    P = create_P(rho, vol)
    weights = diag(P)
    weights_boundary = x[(end - N_boundary + 1):end] # = phi
    function create_D(i)
        sigma = x[(i - 1) * L + 1:(i * L)]
        S = create_S(sigma, N, bandwidth, size_boundary, different_values)
        B = create_B(weights_boundary, normals, on_boundary, i)
        Q = S + B / 2
        D = inv(P) * Q
        return D
    end
    Ds = ntuple(i -> create_D(i), d)
    return weights, weights_boundary, Ds
end

@views function SummationByPartsOperators.multidimensional_optimization_function(x, p)
    V, V_xis, normals, moments, on_boundary, vol, S_cache, A_cache, SV_cache, PV_xi_cache, B_cache, BV_cache, C_cache, VTBV_cache, bandwidth, size_boundary, different_values = p
    S = get_tmp(S_cache, x)
    A = get_tmp(A_cache, x)
    SV = get_tmp(SV_cache, x)
    PV_xi = get_tmp(PV_xi_cache, x)
    B = get_tmp(B_cache, x)
    BV = get_tmp(BV_cache, x)
    VTBV = get_tmp(VTBV_cache, x)
    C = get_tmp(C_cache, x)
    d = length(V_xis)
    N = size(V, 1)
    N_boundary = length(normals)
    L = get_nsigma(N, bandwidth, size_boundary, different_values)
    rho = x[(end - N - N_boundary + 1):(end - N_boundary)]
    phi = x[(end - N_boundary + 1):end]
    P = create_P(rho, vol)
    res = 0.0
    for i in 1:d
        M = moments[i]
        V_xi = V_xis[i]
        sigma = x[(i - 1) * L + 1:(i * L)]
        fill!(S, zero(eltype(S)))
        set_S!(S, sigma, N, bandwidth, size_boundary, different_values)
        fill!(B, zero(eltype(B)))
        set_B!(B, phi, normals, on_boundary, i)
        mul!(SV, S, V)
        mul!(PV_xi, P, V_xi)
        mul!(BV, B, V)
        @. A = SV - PV_xi + 0.5 * BV
        mul!(VTBV, V', BV)
        @. C = VTBV - M
        res += sum(abs2, A) + sum(abs2, C)
    end
    return res
end

function SummationByPartsOperators.function_space_operator(basis_functions, nodes::Vector{T},
                                                           source::SourceOfCoefficients;
                                                           derivative_order = 1, accuracy_order = 0, bandwidth = length(nodes) - 1,
                                                           size_boundary = 2 * bandwidth, different_values = true,
                                                           opt_alg = LBFGS(), options = Options(g_tol = 1e-14, iterations = 10000),
                                                           x0 = nothing, verbose = false) where {T, SourceOfCoefficients}

    if derivative_order != 1
        throw(ArgumentError("Derivative order $derivative_order not implemented."))
    end
    if (length(nodes) < 2 * size_boundary + bandwidth || bandwidth < 1) && (bandwidth != length(nodes) - 1)
        throw(ArgumentError("2 * size_boundary + bandwidth = $(2 * size_boundary + bandwidth) needs to be smaller than or equal to N = $(length(nodes)) and bandwidth = $bandwidth needs to be at least 1."))
    end
    sort!(nodes)
    weights, D = construct_function_space_operator(basis_functions, nodes, source;
                                                   bandwidth, size_boundary, different_values, opt_alg,
                                                   options, x0, verbose)
    return MatrixDerivativeOperator(first(nodes), last(nodes), nodes, weights, D, accuracy_order, source)
end

function inner_H1(f, g, f_derivative, g_derivative, nodes)
    return sum(f.(nodes) .* g.(nodes) + f_derivative.(nodes) .* g_derivative.(nodes))
end
norm_H1(f, f_derivative, nodes) = sqrt(inner_H1(f, f, f_derivative, f_derivative, nodes))

call_orthonormal_basis_function(A, basis_functions, k, x) = sum([basis_functions[i](x)*A[k, i] for i in 1:k])

# This will orthonormalize the basis functions using the Gram-Schmidt process to reduce the condition
# number of the Vandermonde matrix. The matrix A transfers the old basis functions to the new orthonormalized by
# g(x) = A * f(x), where f(x) is the vector of old basis functions and g(x) is the vector of the new orthonormalized
# basis functions. Analogously, we have g'(x) = A * f'(x).
function orthonormalize_gram_schmidt(basis_functions, basis_functions_derivatives, nodes)
    K = length(basis_functions)

    A = LowerTriangular(zeros(eltype(nodes), K, K))

    basis_functions_orthonormalized = Vector{Function}(undef, K)
    basis_functions_orthonormalized_derivatives = Vector{Function}(undef, K)

    for k = 1:K
        A[k, k] = 1
        for j = 1:k-1
            g(x) = call_orthonormal_basis_function(A, basis_functions, j, x)
            g_derivative(x) = call_orthonormal_basis_function(A, basis_functions_derivatives, j, x)
            inner_product = inner_H1(basis_functions[k], g, basis_functions_derivatives[k], g_derivative, nodes)
            norm_squared = inner_H1(g, g, g_derivative, g_derivative, nodes)
            A[k, :] = A[k, :] - inner_product/norm_squared * A[j, :]
        end

        basis_functions_orthonormalized[k] = x -> call_orthonormal_basis_function(A, basis_functions, k, x)
        basis_functions_orthonormalized_derivatives[k] = x -> call_orthonormal_basis_function(A, basis_functions_derivatives, k, x)
        # Normalization
        r = norm_H1(basis_functions_orthonormalized[k], basis_functions_orthonormalized_derivatives[k], nodes)
        A[k, :] = A[k, :] / r
    end
    return basis_functions_orthonormalized, basis_functions_orthonormalized_derivatives
end

function construct_function_space_operator(basis_functions, nodes,
                                           ::GlaubitzNordströmÖffner2023;
                                           bandwidth = length(nodes) - 1,
                                           size_boundary = 2 * bandwidth, different_values = true,
                                           opt_alg = LBFGS(), options = Options(g_tol = 1e-14, iterations = 10000),
                                           x0 = nothing, verbose = false)
    K = length(basis_functions)
    N = length(nodes)
    L = get_nsigma(N, bandwidth, size_boundary, different_values)
    basis_functions_derivatives = [x -> ForwardDiff.derivative(basis_functions[i], x) for i in 1:K]
    basis_functions_orthonormalized, basis_functions_orthonormalized_derivatives = orthonormalize_gram_schmidt(basis_functions,
                                                                                                               basis_functions_derivatives,
                                                                                                               nodes)
    V = vandermonde_matrix(basis_functions_orthonormalized, nodes)
    V_x = vandermonde_matrix(basis_functions_orthonormalized_derivatives, nodes)
    # Here, W satisfies W'*W = I
    # W = [V; -V_x]

    B = spzeros(eltype(nodes), N, N)
    B[1, 1] = -1
    B[N, N] = 1

    R = B * V / 2
    x_length = last(nodes) - first(nodes)
    S = zeros(eltype(nodes), N, N)
    A = zeros(eltype(nodes), N, K)
    SV = zeros(eltype(nodes), N, K)
    PV_x = zeros(eltype(nodes), N, K)
    S_cache = DiffCache(S)
    A_cache = DiffCache(A)
    SV_cache = DiffCache(SV)
    PV_x_cache = DiffCache(PV_x)
    daij_dsigmak = zeros(eltype(nodes), N, K, L)
    daij_drhok = zeros(eltype(nodes), N, K, N)
    p = (V, V_x, R, x_length, S_cache, A_cache, SV_cache, PV_x_cache, bandwidth, size_boundary, different_values, daij_dsigmak, daij_drhok)

    if isnothing(x0)
        x0 = [zeros(L); invsig.(1/N * ones(N))]
    else
        @assert length(x0) == L + N "Initial guess has be L + N = $(L + N) long"
    end

    if bandwidth == N - 1
        fg!(F, G, x) = optimization_function_and_grad!(F, G, x, p)
        result = optimize(Optim.only_fg!(fg!), x0, opt_alg, options)
    else
        f(x) = optimization_function(x, p)
        result = optimize(f, x0, opt_alg, options, autodiff = :forward)
    end
    verbose && display(result)

    x = minimizer(result)
    sigma = x[1:L]
    rho = x[(L + 1):end]
    S = create_S(sigma, N, bandwidth, size_boundary, different_values)
    P = create_P(rho, x_length)
    weights = diag(P)
    Q = S + B/2
    D = inv(P) * Q
    return weights, D
end

@views function optimization_function(x, p)
    V, V_x, R, x_length, S_cache, A_cache, SV_cache, PV_x_cache, bandwidth, size_boundary, different_values = p
    S = get_tmp(S_cache, x)
    A = get_tmp(A_cache, x)
    SV = get_tmp(SV_cache, x)
    PV_x = get_tmp(PV_x_cache, x)
    (N, _) = size(R)
    L = get_nsigma(N, bandwidth, size_boundary, different_values)
    sigma = x[1:L]
    rho = x[(L + 1):end]
    set_S!(S, sigma, N, bandwidth, size_boundary, different_values)
    P = create_P(rho, x_length)
    mul!(SV, S, V)
    mul!(PV_x, P, V_x)
    @. A = SV - PV_x + R
    return norm(A)^2
end

@views function optimization_function_and_grad!(F, G, x, p)
    V, V_x, R, x_length, S_cache, A_cache, SV_cache, PV_x_cache, bandwidth, _, _, daij_dsigmak, daij_drhok = p
    S = get_tmp(S_cache, x)
    A = get_tmp(A_cache, x)
    SV = get_tmp(SV_cache, x)
    PV_x = get_tmp(PV_x_cache, x)
    (N, _) = size(R)
    L = div(N * (N - 1), 2)
    sigma = x[1:L]
    rho = x[(L + 1):end]
    set_S!(S, sigma, N, bandwidth)
    P = create_P(rho, x_length)
    mul!(SV, S, V)
    mul!(PV_x, P, V_x)
    @. A = SV - PV_x + R
    if !isnothing(G)
        fill!(daij_dsigmak, zero(eltype(daij_dsigmak)))
        for k in axes(daij_dsigmak, 3)
            for j in axes(daij_dsigmak, 2)
                for i in axes(daij_dsigmak, 1)
                    l_tilde = k + i - N * (i - 1) + div(i * (i - 1), 2)
                    # same as above, but needs more type conversions
                    # l_tilde = Int(k + i - (i - 1) * (N - i/2))
                    if i + 1 <= l_tilde <= N
                        daij_dsigmak[i, j, k] += V[l_tilde, j]
                    else
                        C = N^2 - 3*N + 2*i - 2*k + 1/4
                        if C >= 0
                            D = sqrt(C)
                            D_plus_one_half = D + 0.5
                            D_plus_one_half_trunc = trunc(D_plus_one_half)
                            if D_plus_one_half == D_plus_one_half_trunc
                                int_D_plus_one_half = trunc(Int, D_plus_one_half_trunc)
                                l_hat = N - int_D_plus_one_half
                                if 1 <= l_hat <= i - 1
                                    daij_dsigmak[i, j, k] -= V[l_hat, j]
                                end
                            end
                        end
                    end
                end
            end
        end
        sig_rho = sig.(rho)
        sig_deriv_rho = sig_deriv.(rho)
        sum_sig_rho = sum(sig_rho)
        for k in axes(daij_drhok, 3)
            for j in axes(daij_drhok, 2)
                for i in axes(daij_drhok, 1)
                    factor1 = x_length * V_x[i, j] / sum_sig_rho^2
                    factor =  factor1 * sig_deriv_rho[k]
                    if k == i
                        daij_drhok[i, j, k] = -factor * (sum_sig_rho - sig_rho[k])
                    else
                        daij_drhok[i, j, k] = factor * sig_rho[i]
                    end
                end
            end
        end
        for k in axes(daij_dsigmak, 3)
            G[k] = 2 * dot(daij_dsigmak[:, :, k], A)
        end
        for k in axes(daij_drhok, 3)
            G[L + k] = 2 * dot(daij_drhok[:, :, k], A)
        end
    end
    if !isnothing(F)
        return norm(A)^2
    end
end

function reconstruct_skew_symmetric!(sigma, S, init_k = 1)
    N = size(S, 1)
    k = init_k
    for i in 1:N
        for j in (i + 1):N
            sigma[k] = S[i, j]
            k += 1
        end
    end
    return k
end

# Only for different_values = true
function reconstruct_banded!(sigma, D, bandwidth, init_k = 1)
    N = size(D, 1)
    k = init_k
    for i in 1:N
        for j in (i + 1):N
            if j - i <= bandwidth
                sigma[k] = D[i, j]
                k += 1
            end
        end
    end
    return k
end

# Only for different_values = true
function reconstruct_triangular!(sigma, C, bandwidth, init_k = 1)
    N = size(C, 1)
    k = init_k
    start_i = N - bandwidth + 1
    for i in start_i:N
        for j in 1:(i - start_i + 1)
            sigma[k] = C[i, j]
            k += 1
        end
    end
    return k
end

# Helper function to get the entries to optimize for from other operators.
# These can, e.g., be used to initialize the optimization problem.
function SummationByPartsOperators.get_optimization_entries(D;
                                                            bandwidth = div(SummationByPartsOperators.accuracy_order(D), 2),
                                                            size_boundary = SummationByPartsOperators.lower_bandwidth(D) + 1,
                                                            different_values = false)
    b = bandwidth
    c = size_boundary
    p = diag(SummationByPartsOperators.mass_matrix(D))
    # for sig = exp this is only equal to the values from the optimization up to a constant, but they give the same P
    # if sig is the logistic function, inverting the normalized logistic function is harder, but this still works
    # (eventhough it is not the exaxt inverse)
    rho = invsig.(p)
    Matrix_D = if D isa MultidimensionalMatrixOperator
        Matrix(D, 1)
    else
        Matrix(D)
    end
    Q = SummationByPartsOperators.mass_matrix(D) * Matrix_D
    S = 0.5 * (Q - Q')
    N = size(D, 1)
    L = get_nsigma(N, b, c, different_values)
    sigma = zeros(L)
    if b == N - 1 # dense operator
        reconstruct_skew_symmetric!(sigma, S)
    else # sparse operator
        if different_values
            k = 1
            # upper left boundary block
            M1 = S[1:c, 1:c]
            k = reconstruct_skew_symmetric!(sigma, M1, k)
            # lower right boundary block
            M2 = S[(N - c + 1):N, (N - c + 1):N]
            k = reconstruct_skew_symmetric!(sigma, M2, k)

            # banded matrix in the middle
            D = S[(c + 1):(N - c), (c + 1):(N - c)]
            k = reconstruct_banded!(sigma, D, b, k)

            # upper central block with triangular part
            C1 = S[1:c, (c + 1):(N - c)]
            k = reconstruct_triangular!(sigma, C1, b, k)
            # central right block with triangular part
            C2 = S[(c + 1):(N - c), (N - c + 1):N]
            k = reconstruct_triangular!(sigma, C2, b, k)
        else
            k = 1
            # upper left boundary block
            M1 = S[1:c, 1:c]
            k = reconstruct_skew_symmetric!(sigma, M1, k)
            # values from triangle block (repeating stencil)
            for i in c:-1:(c - b + 1)
                sigma[k] = S[i, c + 1]
                k += 1
            end
        end
    end
    return [sigma; rho]
end

# This only works if the operator is 1D!
function SummationByPartsOperators.get_multidimensional_optimization_entries(D;
                                                                             bandwidth = div(SummationByPartsOperators.accuracy_order(D), 2),
                                                                             size_boundary = SummationByPartsOperators.lower_bandwidth(D) + 1,
                                                                             different_values = false)
    sigmarho = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
    phi = [1.0, 1.0]
    return [sigmarho; phi]
end

# This only works if the operator is 1D!
function SummationByPartsOperators.get_multidimensional_optimization_entries(D::MultidimensionalMatrixOperator;
                                                                             bandwidth = div(SummationByPartsOperators.accuracy_order(D), 2),
                                                                             size_boundary = SummationByPartsOperators.lower_bandwidth(D) + 1,
                                                                             different_values = false)
    sigmarho = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
    phi = D.weights_boundary
    return [sigmarho; phi]
end

end # module
