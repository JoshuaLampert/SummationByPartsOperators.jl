
"""
    GlaubitzNordströmÖffner2023()

Function space SBP (FSBP) operators given in
- Glaubitz, Nordström, Öffner (2023)
  Summation-by-parts operators for general function spaces.
  SIAM Journal on Numerical Analysis 61, 2, pp. 733-754.

See also
- Glaubitz, Nordström, Öffner (2024)
  An optimization-based construction procedure for function space based
  summation-by-parts operators on arbitrary grids.
  arXiv, arXiv:2405.08770v1.

See [`function_space_operator`](@ref).
"""
struct GlaubitzNordströmÖffner2023 <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzNordströmÖffner2023)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Nordström, Öffner (2023) \n",
              "  Summation-by-parts operators for general function spaces \n",
              "  SIAM Journal on Numerical Analysis 61, 2, pp. 733-754. \n",
              "See also \n",
              "  Glaubitz, Nordström, Öffner (2024) \n",
              "  An optimization-based construction procedure for function \n",
              "    space based summation-by-parts operators on arbitrary grids \n",
              "  arXiv, arXiv:2405.08770v1.")
    end
end

# This function is extended in the package extension SummationByPartsOperatorsOptimExt
"""
    function_space_operator(basis_functions, nodes, source;
                            derivative_order = 1, accuracy_order = 0,
                            bandwidth = length(nodes) - 1, size_boundary = 2 * bandwidth,
                            different_values = true, sparsity_pattern = nothing,
                            opt_alg = Optim.LBFGS(), options = Optim.Options(g_tol = 1e-14, iterations = 10000),
                            autodiff = :forward, x0 = nothing, verbose = false)

Construct an operator that represents a first-derivative operator in a function space spanned by
the `basis_functions`, which is an iterable of functions. The operator is constructed on the
interval `[x_min, x_max]` with the nodes `nodes`, where `x_min` is taken as the minimal value in
`nodes` and `x_max` the maximal value. Note that the `nodes` will be sorted internally. The
`accuracy_order` is the order of the accuracy of the operator, which can optionally be passed,
but does not have any effect on the operator.

The operator is constructed solving an optimization problem with Optim.jl. You can specify the
optimization algorithm, the options for the optimization problem, and the `autodiff` mode with
the keyword arguments `opt_alg`, `options`, and `autodiff` respectively, see also the documentation of
Optim.jl about [configurable options](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
and [automatic differentiation](https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/#Automatic-differentiation).
In this case, reverse mode automatic differentiation is usually significantly faster than forward mode.
We recommend using `autodiff = ADTypes.AutoMooncake(; config = nothing)` or
`autodiff = ADTypes.AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Duplicated)`. Note that
you need to import the package `ADTypes` as well as the corresponding autodiff (i.e., `Mooncake` or `Enzyme`)
package to use these modes.

The initial guess for the optimization problem can be passed with the keyword argument `x0`, which is optional.
If `nothing` is passed, a default initial guess (zeros for the entries of the differentiation matrix and
equal values for all the weights) is used.

There are two alternative ways to enforce sparsity of the resulting operator. The first is by passing
a matrix `sparsity_pattern` that is a matrix of zeros and ones, where the ones indicate the non-zero
entries of the operator. This matrix should be symmetric or `UpperTriangular` and have zeros on the diagonal.

The second way is to use a banded-block structure for the operator as is common, e.g., in finite difference methods.
The keyword arguments `bandwidth` and `size_boundary` specify the bandwidth and the size of the
boundary blocks of the operator, where the default of `bandwidth` is set to `length(nodes) - 1`,
i.e., a dense operator (in this case `size_boundary` is ignored). To construct a sparse operator, you can set the
bandwidth to a smaller value, such that `2 * size_boundary + bandwidth < length(nodes)`, which is a
requirement for the boundary blocks in the upper left and lower right of the resulting operator.
If `different_values` is set to `true` all the entries in the upper right triangle of S (the skew symmetric
part of D) are different, which is generally meaningful for non-equidistant nodes and general bases, if it
is `false` the entries of the stencil are repeated in the central part and the two boundary closures share
their values (makes sense for uniformly distributed nodes and, e.g., a polynomial basis). The keyword
argument `different_values` is ignored for dense operators.

The keyword argument `verbose` can be set to `true` to print information about the optimization process.

The operator that is returned follows the general interface. Currently, it is wrapped in a
[`MatrixDerivativeOperator`](@ref), but this might change in the future.
In order to use this function, the package `Optim` must be loaded.

See also [`GlaubitzNordströmÖffner2023`](@ref).

!!! compat "Julia 1.9"
    This function requires at least Julia 1.9.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function function_space_operator end

"""
    GlaubitzIskeLampertÖffner2024()

Multidimensional function space SBP (MFSBP) operators given in
- Glaubitz, Iske, Lampert, Öffner (2024)
  Efficient construction and application of multi-dimensional summation-by-parts operators
  to global radial basis function methods
  TODO

See [`multidimensional_function_space_operator`](@ref).
"""
struct GlaubitzIskeLampertÖffner2024 <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzIskeLampertÖffner2024)
  if get(io, :compact, false)
    summary(io, source)
  else
      print(io,
          "Glaubitz, Iske, Lampert, Öffner (2024) \n",
          "  Efficient construction and application of multi-dimensional summation-by-parts operators \n",
          "  to global radial basis function methods \n",
          "  TODO.")
  end
end

# This function is extended in the package extension SummationByPartsOperatorsOptimExt
"""
    multidimensional_function_space_operator(basis_functions, nodes, boundary_indices, normals, moments, vol, source;
                                             derivative_order = 1, accuracy_order = 0,
                                             bandwidth = length(nodes) - 1, size_boundary = 2 * bandwidth,
                                             different_values = true, sparsity_pattern = nothing,
                                             opt_alg = Optim.LBFGS(), options = Optim.Options(g_tol = 1e-14, iterations = 10000),
                                             autodiff = :forward, x0 = nothing, verbose = false)

Construct a [`MultidimensionalMatrixOperator`](@ref) that represents a first-derivative operator in a function
space spanned by the `basis_functions`, which is an iterable of functions. The operator is constructed on the
scattered nodes `nodes`. They should be provided as an iterable of `SVector{Dim, T}`. The `boundary_indices`
is a vector of indies that indicates, which nodes are on the boundary. `normals` is a vector of `SVector{Dim, T}`
that contains the normal vectors of the boundary nodes. The `moments` are a `Tuple` of matrices that represent
the moments of the basis functions in each direction. The total volume of the domain is given by `vol`.

The `accuracy_order` is the order of the accuracy of the operator, which can optionally be passed,
but does not have any effect on the operator.

The operator is constructed solving an optimization problem with Optim.jl. You can specify the
optimization algorithm, the options for the optimization problem, and the `autodiff` mode with
the keyword arguments `opt_alg`, `options`, and `autodiff` respectively, see also the documentation of
Optim.jl about [configurable options](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
and [automatic differentiation](https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/#Automatic-differentiation).
In this case, reverse mode automatic differentiation is usually significantly faster than forward mode.
We recommend using `autodiff = ADTypes.AutoMooncake(; config = nothing)` or
`autodiff = ADTypes.AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Duplicated)`. Note that
you need to import the package `ADTypes` as well as the corresponding autodiff (i.e., `Mooncake` or `Enzyme`)
package to use these modes.

The initial guess for the optimization problem can be passed with the keyword argument `x0`, which is optional.
If `nothing` is passed, a default initial guess (zeros for the entries of the differentiation matrix and
equal values for all the weights and boundary weights) is used.

There are two alternative ways to enforce sparsity of the resulting operator. The first is by passing
a matrix `sparsity_pattern` that is a matrix of zeros and ones, where the ones indicate the non-zero
entries of the operator. This matrix should be symmetric or `UpperTriangular` and have zeros on the diagonal.

The second way is to use a banded-block structure for the operator as is common, e.g., in finite difference methods.
The keyword arguments `bandwidth` and `size_boundary` specify the bandwidth and the size of the
boundary blocks of the differentiation matrices in each direction, where the default of `bandwidth`
is set to `length(nodes) - 1`, i.e., dense operators (in this case `size_boundary` is ignored). To
construct sparse operators, you can set the bandwidth to a smaller value, such that
`2 * size_boundary + bandwidth < length(nodes)`, which is a requirement for the boundary blocks in
the upper left and lower right of the resulting operator. If `different_values` is set to `true` all
the entries in the upper right triangle of all matrices S (the skew symmetric parts of the
differentiation matrices D) are different, which is generally meaningful for non-equidistant nodes and
general bases, if it is `false` the entries of the stencil are repeated in the central part and the
two boundary closures share their values (makes sense for uniformly distributed nodes and, e.g., a
polynomial basis). The keyword argument `different_values` is ignored for dense operators.
The parameters `bandwidth`, `size_boundary`, and `different_values` are only used if `sparsity_pattern`
is not provided.

The keyword argument `verbose` can be set to `true` to print information about the optimization process.

In order to use this function, the package `Optim` must be loaded.

See also [`GlaubitzIskeLampertÖffner2024`](@ref).

!!! compat "Julia 1.9"
    This function requires at least Julia 1.9.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function multidimensional_function_space_operator end

function multidimensional_optimization_function end

# Just to be able to call them from outside
function get_optimization_entries end

function get_nsigma end

function get_multidimensional_optimization_entries end

function create_S end
