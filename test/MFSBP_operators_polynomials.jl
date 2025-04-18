using SummationByPartsOperators, Optim
using LinearAlgebra
using StaticArrays
using Test

xmin = -2.0
xmax = 4.0
vol = xmax - xmin
N = 20
verbose = false
opt_kwargs = (;
              options = Optim.Options(f_abstol = 1e-25, g_tol = 1e-16, iterations = 50000),
              opt_alg = BFGS())
atol = 1e-13

# helper functions
function to_S(D::MatrixDerivativeOperator)
    to_S(mass_matrix(D), Matrix(D))
end

function to_S(D::MultidimensionalMatrixDerivativeOperator{1})
    to_S(mass_matrix(D), Matrix(D[1]))
end

function to_S(P, D)
    Q = P * D
    S = 0.5 * (Q - Q')
    return S
end

function to_sigma_rho(x)
    sigma = x[1:(end - N)]
    rho = x[(end - N + 1):end]
    return sigma, rho
end

function compute_boundary_matrix(N)
    B = zeros(N, N)
    B[1, 1] = -1.0
    B[end, end] = 1.0
    return B
end

function compute_moments_boundary(basis_functions, nodes, normals)
    K = length(basis_functions)
    M = zeros(K, K)
    for i in 1:K
        for j in 1:K
            fi = basis_functions[i]
            fj = basis_functions[j]
            M[i, j] = fi(nodes[1]) * fj(nodes[1]) * normals[1][1] +
                      fi(nodes[end]) * fj(nodes[end]) * normals[end][1]
        end
    end
    return (M,)
end

function compute_moments_boundary(functions,
                                  D::SummationByPartsOperators.AbstractMultidimensionalMatrixDerivativeOperator{Dim,
                                                                                                                T}) where {
                                                                                                                           Dim,
                                                                                                                           T
                                                                                                                           }
    K = length(functions)
    nodes = grid(D)
    moments = ntuple(Dim) do i
        M = zeros(T, K, K)
        for k in 1:K
            for l in 1:K
                f = x -> functions[k](x) * functions[l](x)
                M[k, l] = integrate_boundary(f, nodes, D, i)
            end
        end
        M
    end
    return moments
end

function get_sparsity_pattern(D::SummationByPartsOperators.AbstractNonperiodicDerivativeOperator)
    return get_sparsity_pattern(to_S(D))
end

function get_sparsity_pattern(D::SummationByPartsOperators.AbstractMultidimensionalMatrixDerivativeOperator{2})
    P = mass_matrix(D)
    return (get_sparsity_pattern(to_S(P, D[1])), get_sparsity_pattern(to_S(P, D[2])))
end

function get_sparsity_pattern(S)
    return UpperTriangular(S .!= 0.0)
end

function block_banded_sparsity_pattern(N, bandwidth, size_boundary)
    different_values = true
    sigma = ones(SummationByPartsOperators.get_nsigma(N; bandwidth, size_boundary,
                                                      different_values))
    S = SummationByPartsOperators.create_S(sigma, N, bandwidth, size_boundary,
                                           different_values, nothing)
    return get_sparsity_pattern(S)
end

@testset "Reproducing polynomial SBP operators with FSBP operators (1D)" verbose=true begin
    @testset "legendre_derivative_operator (equal)" verbose=true begin
        for n in 2:6
            @testset "n = $n" verbose=true begin
                D_legendre = legendre_derivative_operator(xmin, xmax, n)
                basis = [x -> x^i for i in 0:(n - 1)]
                nodes = collect(grid(D_legendre))

                @testset "function_space_operator" verbose=true begin
                    D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                                                verbose, opt_kwargs...)
                    @test isapprox(Matrix(D), Matrix(D_legendre); atol) # equal
                    @test isapprox(mass_matrix(D), mass_matrix(D_legendre); atol) # equal

                    sparsity_pattern = get_sparsity_pattern(D)
                    D_sparsity_pattern = function_space_operator(basis, nodes,
                                                                 GlaubitzNordströmÖffner2023();
                                                                 sparsity_pattern,
                                                                 verbose, opt_kwargs...)
                    @test isapprox(Matrix(D_sparsity_pattern), Matrix(D_legendre); atol) # equal
                    @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_legendre);
                                   atol) # equal
                end

                @testset "multidimensional_function_space_operator" verbose=true begin
                    basis = [x -> x[1]^i for i in 0:(n - 1)]
                    nodes = SVector.(nodes)
                    boundary_indices_ = [1, n]
                    normals_ = [SVector(-1.0), SVector(1.0)]
                    moments = compute_moments_boundary(basis, nodes, normals_)
                    D = multidimensional_function_space_operator(basis, nodes,
                                                                 boundary_indices_,
                                                                 normals_,
                                                                 moments, vol,
                                                                 GlaubitzIskeLampertÖffner2024();
                                                                 verbose, opt_kwargs...)
                    @test isapprox(Matrix(D[1]), Matrix(D_legendre); atol) # equal
                    @test isapprox(mass_matrix(D), mass_matrix(D_legendre); atol) # equal
                    @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(n);
                                   atol) # equal

                    sparsity_pattern = get_sparsity_pattern(D)
                    sparsity_patterns = (sparsity_pattern,)
                    D_sparsity_pattern = multidimensional_function_space_operator(basis,
                                                                                  nodes,
                                                                                  boundary_indices_,
                                                                                  normals_,
                                                                                  moments,
                                                                                  vol,
                                                                                  GlaubitzIskeLampertÖffner2024();
                                                                                  sparsity_patterns,
                                                                                  verbose,
                                                                                  opt_kwargs...)
                    @test isapprox(Matrix(D_sparsity_pattern[1]), Matrix(D_legendre); atol) # equal
                    @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_legendre);
                                   atol) # equal
                    @test isapprox(mass_matrix_boundary(D_sparsity_pattern, 1),
                                   compute_boundary_matrix(n); atol) # equal
                end
            end
        end
    end

    @testset "MattssonNordström2004" verbose=true begin
        source = MattssonNordström2004()
        for p in (2, 4)
            @testset "p = $p (equal)" verbose=true begin
                D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
                basis = [x -> x^i for i in 0:div(p, 2)]
                nodes = collect(grid(D_poly))
                bandwidth = div(p, 2)
                size_boundary = p
                different_values = false

                @testset "function_space_operator" verbose=true begin
                    D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                                                bandwidth, size_boundary, different_values,
                                                verbose, opt_kwargs...)
                    @test isapprox(Matrix(D), Matrix(D_poly); 1e-12) # equal
                    @test isapprox(mass_matrix(D), mass_matrix(D_poly); 1e-12) # equal
                end

                @testset "multidimensional_function_space_operator" verbose=true begin
                    basis = [x -> x[1]^i for i in 0:div(p, 2)]
                    nodes = SVector.(nodes)
                    boundary_indices_ = [1, N]
                    normals_ = [SVector(-1.0), SVector(1.0)]
                    moments = compute_moments_boundary(basis, nodes, normals_)
                    D = multidimensional_function_space_operator(basis, nodes,
                                                                 boundary_indices_,
                                                                 normals_,
                                                                 moments, vol,
                                                                 GlaubitzIskeLampertÖffner2024();
                                                                 bandwidth, size_boundary,
                                                                 different_values,
                                                                 verbose, opt_kwargs...)
                    @test isapprox(Matrix(D[1]), Matrix(D_poly); atol) # equal
                    @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
                    @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N);
                                   atol) # equal
                end
            end

            @testset "p = $p with different_values = true (almost equal)" verbose=true begin
                D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
                basis = [x -> x^i for i in 0:div(p, 2)]
                nodes = collect(grid(D_poly))
                bandwidth = div(p, 2)
                size_boundary = p
                different_values = true

                @testset "function_space_operator" verbose=true begin
                    D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                                                bandwidth, size_boundary, different_values,
                                                verbose, opt_kwargs...)
                    @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-3) # almost equal
                    @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-4) # almost equal

                    x_poly = SummationByPartsOperators.get_optimization_entries(D_poly;
                                                                                bandwidth,
                                                                                size_boundary,
                                                                                different_values)
                    D_x0 = function_space_operator(basis, nodes,
                                                   GlaubitzNordströmÖffner2023();
                                                   bandwidth, size_boundary,
                                                   different_values,
                                                   verbose, opt_kwargs...,
                                                   x0 = x_poly)
                    @test isapprox(Matrix(D_x0), Matrix(D_poly); atol) # equal
                    @test isapprox(mass_matrix(D_x0), mass_matrix(D_poly); atol) # equal
                    x = SummationByPartsOperators.get_optimization_entries(D_x0; bandwidth,
                                                                           size_boundary,
                                                                           different_values)
                    @test isapprox(x, x_poly; atol)

                    sparsity_pattern = block_banded_sparsity_pattern(N, bandwidth,
                                                                     size_boundary)
                    @test all(get_sparsity_pattern(D) .== sparsity_pattern)

                    D_sparsity_pattern = function_space_operator(basis, nodes,
                                                                 GlaubitzNordströmÖffner2023();
                                                                 sparsity_pattern,
                                                                 verbose, opt_kwargs...)
                    @test isapprox(Matrix(D_sparsity_pattern), Matrix(D_poly); atol = 1e-3) # almost equal
                    @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_poly);
                                   atol = 1e-4) # almost equal

                    # This needs only 1 iteration
                    x_poly_sparsity_pattern = SummationByPartsOperators.get_optimization_entries(D_poly;
                                                                                                 sparsity_pattern)
                    D_sparsity_pattern_x0 = function_space_operator(basis, nodes,
                                                                    GlaubitzNordströmÖffner2023();
                                                                    sparsity_pattern,
                                                                    verbose, opt_kwargs...,
                                                                    x0 = x_poly_sparsity_pattern)
                    @test isapprox(Matrix(D_sparsity_pattern_x0), Matrix(D_poly); atol) # equal
                    @test isapprox(mass_matrix(D_sparsity_pattern_x0), mass_matrix(D_poly);
                                   atol) # equal
                    x = SummationByPartsOperators.get_optimization_entries(D_sparsity_pattern_x0;
                                                                           sparsity_pattern)
                    @test isapprox(x, x_poly_sparsity_pattern; atol)
                end

                @testset "multidimensional_function_space_operator" verbose=true begin
                    basis = [x -> x[1]^i for i in 0:div(p, 2)]
                    nodes = SVector.(nodes)
                    boundary_indices_ = [1, N]
                    normals_ = [SVector(-1.0), SVector(1.0)]
                    moments = compute_moments_boundary(basis, nodes, normals_)
                    D = multidimensional_function_space_operator(basis, nodes,
                                                                 boundary_indices_,
                                                                 normals_,
                                                                 moments, vol,
                                                                 GlaubitzIskeLampertÖffner2024();
                                                                 bandwidth, size_boundary,
                                                                 different_values,
                                                                 verbose, opt_kwargs...)
                    @test isapprox(Matrix(D[1]), Matrix(D_poly); atol = 1e-1) # almost equal
                    @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-2) # almost equal
                    @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N);
                                   atol) # equal

                    x_poly = SummationByPartsOperators.get_multidimensional_optimization_entries(D_poly;
                                                                                                 bandwidth,
                                                                                                 size_boundary,
                                                                                                 different_values)
                    D = multidimensional_function_space_operator(basis, nodes,
                                                                 boundary_indices_,
                                                                 normals_,
                                                                 moments, vol,
                                                                 GlaubitzIskeLampertÖffner2024();
                                                                 bandwidth, size_boundary,
                                                                 different_values,
                                                                 verbose, opt_kwargs...,
                                                                 x0 = x_poly)
                    @test isapprox(Matrix(D[1]), Matrix(D_poly); atol) # equal
                    @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
                    @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N);
                                   atol) # equal
                    x = SummationByPartsOperators.get_multidimensional_optimization_entries(D;
                                                                                            bandwidth,
                                                                                            size_boundary,
                                                                                            different_values)
                    @test isapprox(x, x_poly; atol)

                    sparsity_pattern = block_banded_sparsity_pattern(N, bandwidth,
                                                                     size_boundary)
                    @test all(get_sparsity_pattern(D) .== sparsity_pattern)
                    sparsity_patterns = (sparsity_pattern,)
                    D_sparsity_pattern = multidimensional_function_space_operator(basis,
                                                                                  nodes,
                                                                                  boundary_indices_,
                                                                                  normals_,
                                                                                  moments,
                                                                                  vol,
                                                                                  GlaubitzIskeLampertÖffner2024();
                                                                                  sparsity_patterns,
                                                                                  verbose,
                                                                                  opt_kwargs...)
                    @test isapprox(Matrix(D_sparsity_pattern[1]), Matrix(D_poly);
                                   atol = 1e-1) # almost equal
                    @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_poly);
                                   atol = 1e-2) # almost equal
                    @test isapprox(mass_matrix_boundary(D_sparsity_pattern, 1),
                                   compute_boundary_matrix(N); atol) # equal

                    # This neeeds only 1 iteration
                    x_poly_sparsity_pattern = SummationByPartsOperators.get_multidimensional_optimization_entries(D_poly;
                                                                                                                  sparsity_patterns)
                    D_sparsity_pattern_x0 = multidimensional_function_space_operator(basis,
                                                                                     nodes,
                                                                                     boundary_indices_,
                                                                                     normals_,
                                                                                     moments,
                                                                                     vol,
                                                                                     GlaubitzIskeLampertÖffner2024();
                                                                                     sparsity_patterns,
                                                                                     verbose,
                                                                                     opt_kwargs...,
                                                                                     x0 = x_poly_sparsity_pattern)
                    @test isapprox(Matrix(D_sparsity_pattern_x0[1]), Matrix(D_poly); atol) # equal
                    @test isapprox(mass_matrix(D_sparsity_pattern_x0), mass_matrix(D_poly);
                                   atol) # equal
                    @test isapprox(mass_matrix_boundary(D_sparsity_pattern_x0, 1),
                                   compute_boundary_matrix(N); atol) # equal
                    x = SummationByPartsOperators.get_multidimensional_optimization_entries(D_sparsity_pattern_x0;
                                                                                            sparsity_patterns)
                    @test isapprox(x, x_poly_sparsity_pattern; atol)
                end
            end
        end

        # @testset "p = 6 (not equal)" verbose = true begin
        #     p = 6
        #     D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
        #     basis = [x -> x^i for i in 0:div(p, 2)]
        #     nodes = collect(grid(D_poly))
        #     bandwidth = div(p, 2)
        #     size_boundary = p
        #     different_values = false

        #     @testset "function_space_operator" verbose = true begin
        #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
        #                                     bandwidth, size_boundary, different_values,
        #                                     verbose, opt_kwargs...)
        #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 65.95363413150588; atol) # not equal
        #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal

        #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
        #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
        #                                     bandwidth, size_boundary, different_values,
        #                                     verbose, opt_kwargs...,
        #                                     x0 = x_poly)
        #         @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
        #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
        #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
        #         @test isapprox(x, x_poly; atol)
        #     end

        #     @testset "multidimensional_function_space_operator" verbose = true begin
        #         basis = [x -> x[1]^i for i in 0:div(p, 2)]
        #         nodes = SVector.(nodes)
        #         boundary_indices_ = [1, N]
        #         normals_ = [SVector(-1.0), SVector(1.0)]
        #         moments = compute_moments_boundary(basis, nodes, normals_)
        #         D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
        #                                                      GlaubitzIskeLampertÖffner2024();
        #                                                      bandwidth, size_boundary, different_values,
        #                                                      verbose, opt_kwargs...)
        #         @test isapprox(sum(abs.(Matrix(D[1]) - Matrix(D_poly))), 65.95363412995373; atol) # not equal
        #         @test isapprox(mass_matrix(D), mass_matrix(D_poly)) # equal
        #         @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal

        #         x_poly = SummationByPartsOperators.get_multidimensional_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
        #         D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
        #                                                      GlaubitzIskeLampertÖffner2024();
        #                                                          bandwidth, size_boundary, different_values,
        #                                                          verbose, opt_kwargs...,
        #                                                          x0 = x_poly)
        #             @test isapprox(Matrix(D[1]), Matrix(D_poly); atol) # equal
        #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
        #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal
        #             x = SummationByPartsOperators.get_multidimensional_optimization_entries(D; bandwidth, size_boundary, different_values)
        #             @test isapprox(x, x_poly; atol)
        #         end
        #     end

        #     @testset "p = 6 with different_values = true (not equal)" verbose = true begin
        #         p = 6
        #         D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
        #         basis = [x -> x^i for i in 0:div(p, 2)]
        #         nodes = collect(grid(D_poly))
        #         bandwidth = div(p, 2)
        #         size_boundary = p
        #         different_values = true

        #         @testset "function_space_operator" verbose = true begin
        #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
        #                                         bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
        #             @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 73.46652200054294; atol) # not equal
        #             @test isapprox(sum(abs.(mass_matrix(D) - mass_matrix(D_poly))), 0.9001684652620291; atol) # not equal

        #             x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
        #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
        #                                         bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
        #                                         x0 = x_poly)
        #             @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
        #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
        #             x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
        #             @test isapprox(x, x_poly; atol)
        #         end

        #         @testset "multidimensional_function_space_operator" verbose = true begin
        #             basis = [x -> x[1]^i for i in 0:div(p, 2)]
        #             nodes = SVector.(nodes)
        #             boundary_indices_ = [1, N]
        #             normals_ = [SVector(-1.0), SVector(1.0)]
        #             moments = compute_moments_boundary(basis, nodes, normals_)
        #             D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
        #                                                          GlaubitzIskeLampertÖffner2024();
        #                                                          bandwidth, size_boundary, different_values,
        #                                                          verbose, opt_kwargs...)
        #             @test isapprox(sum(abs.(Matrix(D[1]) - Matrix(D_poly))), 110.71611497448345; atol) # not equal
        #             @test isapprox(sum(abs.(mass_matrix(D) - mass_matrix(D_poly))), 1.3990176657854683; atol) # not equal
        #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal

        #             x_poly = SummationByPartsOperators.get_multidimensional_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
        #             D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
        #                                                          GlaubitzIskeLampertÖffner2024();
        #                                                          bandwidth, size_boundary, different_values,
        #                                                          verbose, opt_kwargs...,
        #                                                          x0 = x_poly)
        #             @test isapprox(Matrix(D[1]), Matrix(D_poly); atol) # equal
        #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
        #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal
        #             x = SummationByPartsOperators.get_multidimensional_optimization_entries(D; bandwidth, size_boundary, different_values)
        #             @test isapprox(x, x_poly; atol)
        #         end
        #     end
    end

    # @testset "MattssonAlmquistVanDerWeide2018Minimal" verbose = true begin
    #     source = MattssonAlmquistVanDerWeide2018Minimal()
    #     @testset "p = 4 (equal)" verbose = true begin
    #         p = 4
    #         D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
    #         basis = [x -> x^i for i in 0:div(p, 2)]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = div(p, 2)
    #         size_boundary = p
    #         different_values = false

    #         @testset "function_space_operator" verbose = true begin
    #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #                                         bandwidth, size_boundary, different_values,
    #                                         verbose, opt_kwargs...)
    #             @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-10) # equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-12) # equal
    #         end

    #         @testset "multidimensional_function_space_operator" verbose = true begin
    #             basis = [x -> x[1]^i for i in 0:div(p, 2)]
    #             nodes = SVector.(nodes)
    #             boundary_indices_ = [1, N]
    #             normals_ = [SVector(-1.0), SVector(1.0)]
    #             moments = compute_moments_boundary(basis, nodes, normals_)
    #             D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
    #                                                          GlaubitzIskeLampertÖffner2024();
    #                                                          bandwidth, size_boundary, different_values,
    #                                                          verbose, opt_kwargs...)
    #             @test isapprox(Matrix(D[1]), Matrix(D_poly); 1e-12) # equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); 1e-12) # equal
    #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal
    #         end
    #     end

    #     @testset "p = 4 with different_values = true (almost equal)" verbose = true begin
    #         p = 4
    #         D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
    #         basis = [x -> x^i for i in 0:div(p, 2)]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = div(p, 2)
    #         size_boundary = p
    #         different_values = true

    #         @testset "function_space_operator" verbose = true begin
    #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #                                         bandwidth, size_boundary, different_values,
    #                                         verbose, opt_kwargs...)
    #             @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-3) # almost equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-4) # almost equal

    #             x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
    #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #                                         bandwidth, size_boundary, different_values,
    #                                         verbose, opt_kwargs...,
    #                                         x0 = x_poly)
    #             @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-11) # equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #             x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
    #             @test isapprox(x, x_poly; atol = 1e-12)
    #         end

    #         @testset "multidimensional_function_space_operator" verbose = true begin
    #             basis = [x -> x[1]^i for i in 0:div(p, 2)]
    #             nodes = SVector.(nodes)
    #             boundary_indices_ = [1, N]
    #             normals_ = [SVector(-1.0), SVector(1.0)]
    #             moments = compute_moments_boundary(basis, nodes, normals_)
    #             D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
    #                                                         GlaubitzIskeLampertÖffner2024();
    #                                                         bandwidth, size_boundary, different_values,
    #                                                         verbose, opt_kwargs...)
    #             @test isapprox(Matrix(D[1]), Matrix(D_poly); atol = 1.0) # almost equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-1) # almost equal
    #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal

    #             x_poly = SummationByPartsOperators.get_multidimensional_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
    #             D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
    #                                                         GlaubitzIskeLampertÖffner2024();
    #                                                         bandwidth, size_boundary, different_values,
    #                                                         verbose, opt_kwargs...,
    #                                                         x0 = x_poly)
    #             @test isapprox(Matrix(D[1]), Matrix(D_poly); 1e-12) # equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); 1e-12) # equal
    #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal
    #             x = SummationByPartsOperators.get_multidimensional_optimization_entries(D; bandwidth, size_boundary, different_values)
    #             @test isapprox(x, x_poly; 1e-12)
    #         end
    #     end

    #     @testset "p = 6 (not equal)" verbose = true begin
    #         p = 6
    #         D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
    #         basis = [x -> x^i for i in 0:div(p, 2)]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = div(p, 2)
    #         size_boundary = p
    #         different_values = false

    #         @testset "function_space_operator" verbose = true begin
    #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #                                         bandwidth, size_boundary, different_values,
    #                                         verbose, opt_kwargs...)
    #             @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 67.83380576402465; atol) # not equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal

    #             x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #                                         bandwidth, size_boundary, different_values,
    #                                         verbose, opt_kwargs...,
    #                                         x0 = x_poly)
    #             @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-12) # equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #             x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #             @test isapprox(x, x_poly; atol)
    #         end

    #         @testset "multidimensional_function_space_operator" verbose = true begin
    #             basis = [x -> x[1]^i for i in 0:div(p, 2)]
    #             nodes = SVector.(nodes)
    #             boundary_indices_ = [1, N]
    #             normals_ = [SVector(-1.0), SVector(1.0)]
    #             moments = compute_moments_boundary(basis, nodes, normals_)
    #             D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
    #                                                          GlaubitzIskeLampertÖffner2024();
    #                                                          bandwidth, size_boundary, different_values,
    #                                                          verbose, opt_kwargs...)
    #             @test isapprox(sum(abs.(Matrix(D[1]) - Matrix(D_poly))), 67.83380574798099; atol) # not equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal

    #             x_poly = SummationByPartsOperators.get_multidimensional_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
    #             D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
    #                                                          GlaubitzIskeLampertÖffner2024();
    #                                                          bandwidth, size_boundary, different_values,
    #                                                          verbose, opt_kwargs...,
    #                                                          x0 = x_poly)
    #             @test isapprox(Matrix(D[1]), Matrix(D_poly); atol) # equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal
    #             x = SummationByPartsOperators.get_multidimensional_optimization_entries(D; bandwidth, size_boundary, different_values)
    #             @test isapprox(x, x_poly; atol)
    #         end
    #     end

    #     @testset "p = 6 with different_values = true (not equal)" verbose = true begin
    #         p = 6
    #         D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
    #         basis = [x -> x^i for i in 0:div(p, 2)]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = div(p, 2)
    #         size_boundary = p
    #         different_values = true

    #         @testset "function_space_operator" verbose = true begin
    #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #                                         bandwidth, size_boundary, different_values,
    #                                         verbose, opt_kwargs...)
    #             @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 84.3543668148164; atol) # not equal
    #             @test isapprox(sum(abs.(mass_matrix(D) - mass_matrix(D_poly))), 0.8601601363436975; atol) # not equal

    #             x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
    #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #                                         bandwidth, size_boundary, different_values,
    #                                         verbose, opt_kwargs...,
    #                                         x0 = x_poly)
    #             @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-12) # equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #             x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
    #             @test isapprox(x, x_poly; atol)
    #         end

    #         @testset "multidimensional_function_space_operator" verbose = true begin
    #             basis = [x -> x[1]^i for i in 0:div(p, 2)]
    #             nodes = SVector.(nodes)
    #             boundary_indices_ = [1, N]
    #             normals_ = [SVector(-1.0), SVector(1.0)]
    #             moments = compute_moments_boundary(basis, nodes, normals_)
    #             D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
    #                                                          GlaubitzIskeLampertÖffner2024();
    #                                                          bandwidth, size_boundary, different_values,
    #                                                          verbose, opt_kwargs...)
    #             @test isapprox(sum(abs.(Matrix(D[1]) - Matrix(D_poly))), 97.09483989061656; atol) # not equal
    #             @test isapprox(sum(abs.(mass_matrix(D) - mass_matrix(D_poly))), 1.5677588546370986; atol) # not equal
    #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal

    #             x_poly = SummationByPartsOperators.get_multidimensional_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
    #             D = multidimensional_function_space_operator(basis, nodes, boundary_indices_, normals_, moments, vol,
    #                                                          GlaubitzIskeLampertÖffner2024();
    #                                                          bandwidth, size_boundary, different_values,
    #                                                          verbose, opt_kwargs...,
    #                                                          x0 = x_poly)
    #             @test isapprox(Matrix(D[1]), Matrix(D_poly); atol) # equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #             @test isapprox(mass_matrix_boundary(D, 1), compute_boundary_matrix(N); atol) # equal
    #             x = SummationByPartsOperators.get_multidimensional_optimization_entries(D; bandwidth, size_boundary, different_values)
    #             @test isapprox(x, x_poly; atol)
    #         end
    #     end
    # end

    # @testset "MattssonAlmquistVanDerWeide2018Accurate" verbose = true begin
    #     source = MattssonAlmquistVanDerWeide2018Accurate()
    #     @testset "p = 4 (equal)" verbose = true begin
    #         p = 4
    #         D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
    #         basis = [x -> x^i for i in 0:div(p, 2)]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = div(p, 2)
    #         size_boundary = p
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-12) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #     end

    #     @testset "p = 4 with different_values = true (almost equal)" verbose = true begin
    #         p = 4
    #         D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
    #         basis = [x -> x^i for i in 0:div(p, 2)]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = div(p, 2)
    #         size_boundary = p
    #         different_values = true

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-3) # almost equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-4) # almost equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-12) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
    #         @test isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 6 (not equal)" verbose = true begin
    #         p = 6
    #         D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
    #         basis = [x -> x^i for i in 0:div(p, 2)]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = div(p, 2)
    #         size_boundary = p
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 73.70761739735096; atol) # not equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-12) # equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-11) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test isapprox(x, x_poly; atol = 1e-12)
    #     end

    #     @testset "p = 6 with different_values = true (not equal)" verbose = true begin
    #         p = 6
    #         D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
    #         basis = [x -> x^i for i in 0:div(p, 2)]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = div(p, 2)
    #         size_boundary = p
    #         different_values = true

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 89.7912068831487; atol) # not equal
    #         @test isapprox(sum(abs.(mass_matrix(D) - mass_matrix(D_poly))), 0.9355094170583464; atol) # not equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-11) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
    #         @test isapprox(x, x_poly; atol = 1e-12)
    #     end
    # end

    # # Don't test different_values = true for the MattssonAlmquistCarpenter2014 operators for simplicity
    # @testset "MattssonAlmquistCarpenter2014Extended (not equal)" verbose = true begin
    #     source = MattssonAlmquistCarpenter2014Extended()
    #     @testset "p = 2, degree up to 1" begin
    #         D_poly = derivative_operator(source, 1, 2, xmin, xmax, N)
    #         basis = [one, identity]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 1
    #         size_boundary = 3
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 14.295867819799426; atol) # not equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 2, degree up to 2" begin
    #         D_poly = derivative_operator(source, 1, 2, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 1
    #         size_boundary = 3
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 1.0827534543081196; atol) # not equal, but closer than degree up to 1

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 1.0827534543078705; atol) # not equal, same solution as above
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test !isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 4, degree up to 2" begin
    #         D_poly = derivative_operator(source, 1, 4, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 2
    #         size_boundary = 6
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 51.04863302117427; atol) # not equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 4, degree up to 3" begin
    #         D_poly = derivative_operator(source, 1, 4, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2, x -> x^3]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 2
    #         size_boundary = 6
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 17.64645044509875; atol) # not equal, but closer than degree up to 2

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 0.15237872690570192; atol) # not equal, but very close
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test !isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 6, degree up to 3" begin
    #         D_poly = derivative_operator(source, 1, 6, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2, x -> x^3]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 3
    #         size_boundary = 8
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 72.25472510914369; atol) # not equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 6, degree up to 4" begin
    #         D_poly = derivative_operator(source, 1, 6, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2, x -> x^3, x -> x^4]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 3
    #         size_boundary = 8
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 7.770755805130392; atol) # not equal, but closer than degree up to 3

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 0.10011150654961294; atol) # not equal, but very close
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test !isapprox(x, x_poly; atol)
    #     end
    # end

    # @testset "MattssonAlmquistCarpenter2014Optimal (not equal)" verbose = true begin
    #     source = MattssonAlmquistCarpenter2014Optimal()
    #     @testset "p = 2, degree up to 1" begin
    #         D_poly = derivative_operator(source, 1, 2, xmin, xmax, N)
    #         basis = [one, identity]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 1
    #         size_boundary = 3
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 15.37970781810084; atol) # not equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 2, degree up to 2" begin
    #         D_poly = derivative_operator(source, 1, 2, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 1
    #         size_boundary = 3
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 1.0923973100891688; atol) # not equal, but closer than degree up to 1

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 1.092397310086492; atol) # not equal, same solution as above
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test !isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 4, degree up to 2" begin
    #         D_poly = derivative_operator(source, 1, 4, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 2
    #         size_boundary = 6
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 86.22100592201839; atol) # not equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 4, degree up to 3" begin
    #         D_poly = derivative_operator(source, 1, 4, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2, x -> x^3]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 2
    #         size_boundary = 6
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 53.43301616180371; atol) # not equal, but closer than degree up to 2

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 0.4884165485425883; atol) # not equal, but very close
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test !isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 6, degree up to 3" begin
    #         D_poly = derivative_operator(source, 1, 6, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2, x -> x^3]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 3
    #         size_boundary = 8
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 131.56026715643415; atol) # not equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test isapprox(x, x_poly; atol)
    #     end

    #     @testset "p = 6, degree up to 4" begin
    #         D_poly = derivative_operator(source, 1, 6, xmin, xmax, N)
    #         basis = [one, identity, x -> x^2, x -> x^3, x -> x^4]
    #         nodes = collect(grid(D_poly))
    #         bandwidth = 3
    #         size_boundary = 8
    #         different_values = false

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 98.36506424063853; atol) # not equal, but closer than degree up to 3

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 0.5663315024377179; atol) # not equal, but very close
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test !isapprox(x, x_poly; atol)
    #     end
    # end
end

ymin = -2.0
ymax = 1.0
N_x = 10
N_y = 9
@testset "Reproducing polynomial tensor product SBP operators with MFSBP operators (2D)" verbose=true begin
    for p in (2, 4)
        @testset "p = $p (equal)" verbose=true begin
            D_1 = derivative_operator(MattssonNordström2004(), derivative_order = 1,
                                      accuracy_order = p,
                                      xmin = xmin, xmax = xmax, N = N_x)
            D_2 = derivative_operator(MattssonNordström2004(), derivative_order = 1,
                                      accuracy_order = p,
                                      xmin = ymin, xmax = ymax, N = N_y)

            D_t = tensor_product_operator_2D(D_1, D_2)

            basis = Function[x -> x[1]^i * x[2]^j for i in 0:div(p, 2), j in 0:div(p, 2)]
            nodes = grid(D_t)
            boundary_indices_ = boundary_indices(D_t)
            normals_ = normals(D_t)
            moments = compute_moments_boundary(basis, D_t)
            vol = (xmax - xmin) * (ymax - ymin)
            sparsity_patterns = get_sparsity_pattern(D_t)

            corners_x_dir = [N_y + 1, # lower left corner
                N_x + N_y, # upper left corner
                N_x + N_y + 1, # lower right corner
                2 * N_x + N_y] # upper right corner
            corners_y_dir = [1, # lower left corner
                N_y,  # upper left corner
                2 * N_x + N_y + 1, # lower right corner
                2 * (N_x + N_y)] # upper right corner
            corners = (corners_x_dir, corners_y_dir)

            D = multidimensional_function_space_operator(basis,
                                                         nodes,
                                                         boundary_indices_,
                                                         normals_,
                                                         moments,
                                                         vol,
                                                         GlaubitzIskeLampertÖffner2024();
                                                         sparsity_patterns,
                                                         corners,
                                                         verbose,
                                                         opt_kwargs...)

            D_t_MFSBP = TensorProductOperator(D, N_x, N_y)
            atol = p == 2 ? 1e-11 : 1e1 # almost equal (some tests also pass with lower `atol`)
            @test isapprox(Matrix(D_t_MFSBP[1]), Matrix(D_t[1]); atol) # equal
            @test isapprox(Matrix(D_t_MFSBP[2]), Matrix(D_t[2]); atol) # equal
            @test isapprox(mass_matrix(D_t_MFSBP), mass_matrix(D_t); atol) # equal
            @test isapprox(mass_matrix_boundary(D_t_MFSBP, 1),
                           mass_matrix_boundary(D_t, 1); atol) # equal
            @test isapprox(mass_matrix_boundary(D_t_MFSBP, 2),
                           mass_matrix_boundary(D_t, 2); atol) # equal

            # This neeeds only 1 (or 2) iteration(s)
            x_tensor = SummationByPartsOperators.get_multidimensional_optimization_entries(D_t;
                                                                                           sparsity_patterns)

            D_x0 = multidimensional_function_space_operator(basis,
                                                            nodes,
                                                            boundary_indices_,
                                                            normals_,
                                                            moments,
                                                            vol,
                                                            GlaubitzIskeLampertÖffner2024();
                                                            sparsity_patterns,
                                                            corners,
                                                            verbose,
                                                            opt_kwargs...,
                                                            x0 = x_tensor)

            D_t_x0 = TensorProductOperator(D_x0, N_x, N_y)
            atol = 1e-13
            @test isapprox(Matrix(D_t_x0[1]), Matrix(D_t[1]); atol) # equal
            @test isapprox(Matrix(D_t_x0[2]), Matrix(D_t[2]); atol) # equal
            @test isapprox(mass_matrix(D_t_x0), mass_matrix(D_t); atol) # equal
            @test isapprox(mass_matrix_boundary(D_t_x0, 1),
                           mass_matrix_boundary(D_t, 1); atol) # equal
            @test isapprox(mass_matrix_boundary(D_t_x0, 2),
                           mass_matrix_boundary(D_t, 2); atol) # equal
            x = SummationByPartsOperators.get_multidimensional_optimization_entries(D_t_x0;
                                                                                    sparsity_patterns)
            @test isapprox(x, x_tensor; atol)
        end
    end
end

nothing # to suppress final output
