using SummationByPartsOperators, Optim
using Test

xmin = -2.0
xmax = 4.0
N = 20
verbose = false
opt_kwargs = (; options=Optim.Options(g_tol=1e-16, iterations=50000), opt_alg=BFGS())
atol = 1e-13

# helper functions
function to_S(D)
    Q = mass_matrix(D) * Matrix(D)
    S = 0.5 * (Q - Q')
    return S
end

function to_sigma_rho(x)
    sigma = x[1:(end-N)]
    rho = x[(end-N+1):end]
    return sigma, rho
end

@testset "Reproducing polynomials with FSBP operators" verbose = true begin
    # @testset "legendre_derivative_operator (equal)" verbose = true begin
    #     for n in 2:6
    #         @testset "n = $n" verbose = true begin
    #             D_legendre = legendre_derivative_operator(xmin, xmax, n)
    #             basis = [x -> x^i for i in 0:(n-1)]
    #             nodes = collect(grid(D_legendre))

    #             D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #                                         verbose, opt_kwargs...)
    #             @test isapprox(Matrix(D), Matrix(D_legendre); atol) # equal
    #             @test isapprox(mass_matrix(D), mass_matrix(D_legendre); atol) # equal
    #         end
    #     end
    # end

    @testset "MattssonNordström2004" verbose = true begin
        source = MattssonNordström2004()
        for p in [2, 4]
            @testset "p = $p (equal)" verbose = true begin
                D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
                basis = [x -> x^i for i in 0:div(p, 2)]
                nodes = collect(grid(D_poly))
                bandwidth = div(p, 2)
                size_boundary = p
                different_values = false

                D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                    bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
                @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
            end

            @testset "p = $p with different_values = true (almost equal)" verbose = true begin
                D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
                basis = [x -> x^i for i in 0:div(p, 2)]
                nodes = collect(grid(D_poly))
                bandwidth = div(p, 2)
                size_boundary = p
                different_values = true

                D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                    bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
                @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-3) # almost equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-4) # almost equal

                x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
                D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                    bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
                    x0=x_poly)
                @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
                x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
                @test isapprox(x, x_poly; atol)
            end
        end

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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 65.95363412753869; atol) # not equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test isapprox(x, x_poly; atol)
        end

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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 73.70790308476734; atol) # not equal
    #         @test isapprox(sum(abs.(mass_matrix(D) - mass_matrix(D_poly))), 0.9211554318326064; atol) # not equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
    #         @test isapprox(x, x_poly; atol)
    #     end
    # end

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

    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-10) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-12) # equal
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
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-11) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
    #         @test isapprox(x, x_poly; atol = 1e-12)
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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 67.83380575510824; atol) # not equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-12) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test isapprox(x, x_poly; atol)
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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 84.9552535991595; atol) # not equal
    #         @test isapprox(sum(abs.(mass_matrix(D) - mass_matrix(D_poly))), 0.8969250760152074; atol) # not equal

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary, different_values)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-12) # equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary, different_values)
    #         @test isapprox(x, x_poly; atol)
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
    #         @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-4) # almost equal
    #         @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-5) # almost equal

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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 73.70761740012675; atol) # not equal
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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 89.79013292902668; atol) # not equal
    #         @test isapprox(sum(abs.(mass_matrix(D) - mass_matrix(D_poly))), 0.9354358447938111; atol) # not equal

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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 1.0827534543090689; atol) # not equal, but closer than degree up to 1

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 1.0827534543075044; atol) # not equal, same solution as above
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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 51.04866053502858; atol) # not equal

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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 17.646450445474215; atol) # not equal, but closer than degree up to 2

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 0.1523787269058588; atol) # not equal, but very close
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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 72.2547280860491; atol) # not equal

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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 7.7707557935451135; atol) # not equal, but closer than degree up to 3

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 0.10011150654335293; atol) # not equal, but very close
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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 1.0923973100889313; atol) # not equal, but closer than degree up to 1

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 1.0923973100884372; atol) # not equal, same solution as above
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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 86.2210237299501; atol) # not equal

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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 53.433016161721525; atol) # not equal, but closer than degree up to 2

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 0.4884165485409877; atol) # not equal, but very close
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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 131.5602669672018; atol) # not equal

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
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 98.36506424140092; atol) # not equal, but closer than degree up to 3

    #         x_poly = SummationByPartsOperators.get_optimization_entries(D_poly; bandwidth, size_boundary)
    #         D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
    #             bandwidth, size_boundary, different_values, verbose, opt_kwargs...,
    #             x0=x_poly)
    #         @test isapprox(sum(abs.(Matrix(D) - Matrix(D_poly))), 0.5663315024415888; atol) # not equal, but very close
    #         x = SummationByPartsOperators.get_optimization_entries(D; bandwidth, size_boundary)
    #         @test !isapprox(x, x_poly; atol)
    #     end
    # end
end

nothing # to suppress final output
