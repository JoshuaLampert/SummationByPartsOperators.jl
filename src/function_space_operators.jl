
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
                            derivative_order = 1, accuracy_order = 0, bandwidth = length(nodes) - 1,
                            size_boundary = 2 * bandwidth, different_values = true,
                            opt_alg = Optim.LBFGS(), options = Optim.Options(g_tol = 1e-14, iterations = 10000),
                            x0 = nothing, verbose = false)

Construct an operator that represents a first-derivative operator in a function space spanned by
the `basis_functions`, which is an iterable of functions. The operator is constructed on the
interval `[x_min, x_max]` with the nodes `nodes`, where `x_min` is taken as the minimal value in
`nodes` and `x_max` the maximal value. Note that the `nodes` will be sorted internally. The
`accuracy_order` is the order of the accuracy of the operator, which can optionally be passed,
but does not have any effect on the operator.

The operator is constructed solving an optimization problem with Optim.jl. You can specify the
optimization algorithm and options for the optimization problem with the keyword arguments `opt_alg`
and `options` respectively, see also the
[documentation of Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/user/config/). The initial
guess for the optimization problem can be passed with the keyword argument `x0`, which is optional.
If `nothing` is passed, a default initial guess (zeros for the entries of the differentiation matrix and
equal values for all the weights) is used.

The keyword arguments `bandwidth` and `size_boundary` specifiy the bandwidth and the size of the
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

function get_optimization_entries end

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
          "  to global radial basis function methodss \n",
          "  TODO.")
  end
end

"""
  MultidimensionalFunctionSpaceOperator{Dim, T}

Multidimensional function space operator that represents a first-derivative operator in a function space.

To be obtain the derivative operator in a specific direction, use `Matrix(D, dim)`. The boundary operator in
a specific direction can be obtained with `mass_matrix_boundary(D, dim)`.

See also [`multidimensional_function_space_operator`](@ref), [`GlaubitzIskeLampertÖffner2024`](@ref).
"""
@auto_hash_equals struct MultidimensionalFunctionSpaceOperator{Dim, T, NodesType, DType <: AbstractMatrix{T}, SourceOfCoefficients} <: AbstractNonperiodicDerivativeOperator{T}
    grid::NodesType # length(grid) == N, e.g. Vector{SVector{Dim, T}} or `NodeSet` from KernelInterpolation.jl
    on_boundary::Vector{Bool} # length(on_boundary) == N
    normals::Vector{SVector{Dim, T}} # length(normals) == N_boundary < N
    weights::Vector{T} # length(weights) == N
    weights_boundary::Vector{T} # length(weights_boundary) == N_boundary < N
    Ds::NTuple{Dim, DType}
    accuracy_order::Int
    source::SourceOfCoefficients

    function MultidimensionalFunctionSpaceOperator(nodes::NodesType,
                                                   on_boundary::Vector{Bool},
                                                   normals::Vector{SVector{Dim, T}},
                                                   weights::Vector{T}, weights_boundary::Vector{T},
                                                   Ds::NTuple{Dim, DType}, accuracy_order::Int,
                                                   source::SourceOfCoefficients) where {Dim, T <: Real, NodesType, DType <: AbstractMatrix{T}, SourceOfCoefficients}
        new{Dim, T, NodesType, DType, SourceOfCoefficients}(nodes, on_boundary, normals, weights, weights_boundary, Ds, accuracy_order, source)
    end
end

Base.ndims(::MultidimensionalFunctionSpaceOperator{Dim}) where {Dim} = Dim
derivative_order(::MultidimensionalFunctionSpaceOperator) = 1
Matrix(D::MultidimensionalFunctionSpaceOperator, dim::Int) = Matrix(D.Ds[dim])

source_of_coefficients(D::MultidimensionalFunctionSpaceOperator) = D.source

function integrate(func, u, D::MultidimensionalFunctionSpaceOperator)
  return integrate(func, u, D.weights)
end

function integrate_boundary(func, u, D::MultidimensionalFunctionSpaceOperator, dim)
  return integrate(func, u, weights_boundary_scaled(D, dim))
end

mass_matrix(D::MultidimensionalFunctionSpaceOperator) = Diagonal(D.weights)
# TODO: more efficient
weights_boundary(D::MultidimensionalFunctionSpaceOperator) = get_weight_boundary.(Ref(D), 1:length(grid(D)))
weights_boundary_scaled(D::MultidimensionalFunctionSpaceOperator, dim::Int) = get_weight_boundary_scaled.(Ref(D), Ref(dim), 1:length(grid(D)))
mass_matrix_boundary(D::MultidimensionalFunctionSpaceOperator, dim::Int) = Diagonal(weights_boundary_scaled(D, dim))

Base.eltype(::MultidimensionalFunctionSpaceOperator{Dim, T}) where {Dim, T} = T

function scale_by_mass_matrix!(u::AbstractVector, D::MultidimensionalFunctionSpaceOperator, factor=true)
  N, _ = size(D)
  @boundscheck begin
      @argcheck N == length(u)
  end

  @inbounds @simd for i in eachindex(u, D.weights)
      u[i] = factor * u[i] * D.weights[i]
  end

  return u
end

function scale_by_inverse_mass_matrix!(u::AbstractVector, D::MultidimensionalFunctionSpaceOperator, factor=true)
  N, _ = size(D)
  @boundscheck begin
      @argcheck N == length(u)
  end

  @inbounds @simd for i in eachindex(u, D.weights)
      u[i] = factor * u[i] / D.weights[i]
  end

  u
end

function get_weight(D::MultidimensionalFunctionSpaceOperator, i::Int)
  @unpack weights = D
  N, _ = size(D)
  @boundscheck begin
      @argcheck 1 <= i <= N
  end
  @inbounds ω = weights[i]
  ω
end

function get_weight_boundary(D::MultidimensionalFunctionSpaceOperator, i::Int)
  @unpack weights_boundary, on_boundary = D
  N, _ = size(D)
  @boundscheck begin
      @argcheck 1 <= i <= N
  end
  if !on_boundary[i]
      return zero(eltype(D))
  end
  j = sum(view(on_boundary, 1:i))
  @inbounds ω = weights_boundary[j]
  return ω
end

function get_weight_boundary_scaled(D::MultidimensionalFunctionSpaceOperator, dim::Int, i::Int)
  @unpack normals, on_boundary = D
  if !on_boundary[i]
    return zero(eltype(D))
  end
  ω = get_weight_boundary(D, i)
  j = sum(view(on_boundary, 1:i))
  ω * normals[j][dim]
end

function Base.show(io::IO, D::MultidimensionalFunctionSpaceOperator)
  if get(io, :compact, false)
      summary(io, D)
  else
      x = grid(D)
      print(io, ndims(D), "-dimensional function space first-derivative operator {T=", eltype(D), "}")
      print(io, " on ", length(x), " nodes")
  end
end

# TODO: mul! How? Depends on direction

function lower_bandwidth(D::MultidimensionalFunctionSpaceOperator)
  size(D, 1) - 1
end

function upper_bandwidth(D::MultidimensionalFunctionSpaceOperator)
  size(D, 1) - 1
end

function accuracy_order(D::MultidimensionalFunctionSpaceOperator)
  D.accuracy_order
end

# TODO
function left_boundary_weight(D::MultidimensionalFunctionSpaceOperator)
  @inbounds retval = D.weights[begin]
  retval
end

function right_boundary_weight(D::MultidimensionalFunctionSpaceOperator)
  @inbounds retval = D.weights[end]
  retval
end

# This function is extended in the package extension SummationByPartsOperatorsOptimExt
"""
    multidimensional_function_space_operator(basis_functions, nodes, on_boundary, normals, moments, vol, source;
                                             derivative_order = 1, accuracy_order = 0,
                                             bandwidth = length(nodes) - 1, size_boundary = 2 * bandwidth,
                                             different_values = true,
                                             opt_alg = Optim.LBFGS(), options = Optim.Options(g_tol = 1e-14, iterations = 10000),
                                             x0 = nothing, verbose = false)

Construct a multidimensional operator that represents a first-derivative operator in a function
space spanned by the `basis_functions`, which is an iterable of functions. The operator is constructed on the
scattered nodes `nodes`. They should be provided as an iterable of `SVector{Dim, T}`. The `on_boundary`
is an iterable of booleans that indicates whether a node is on the boundary or not. `normals` is a
vector of `SVector{Dim, T}` that contains the normal vectors of the boundary nodes. The `moments` are a
`Tuple` of matrices that represent the moments of the basis functions in each direction. The total volume of the
domain is given by `vol`.

The `accuracy_order` is the order of the accuracy of the operator, which can optionally be passed,
but does not have any effect on the operator.

The operator is constructed solving an optimization problem with Optim.jl. You can specify the
optimization algorithm and options for the optimization problem with the keyword arguments `opt_alg`
and `options` respectively, see also the
[documentation of Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/user/config/). The initial
guess for the optimization problem can be passed with the keyword argument `x0`, which is optional.
If `nothing` is passed, a default initial guess (zeros for the entries of the differentiation matrix and
equal values for all the weights and boundary weights) is used.

The keyword arguments `bandwidth` and `size_boundary` specifiy the bandwidth and the size of the
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

The keyword argument `verbose` can be set to `true` to print information about the optimization process.

The operator that is a [`MultidimensionalFunctionSpaceOperator`](@ref) and follows the general interface.
In order to use this function, the package `Optim` must be loaded.

See also [`GlaubitzIskeLampertÖffner2024`](@ref).

!!! compat "Julia 1.9"
    This function requires at least Julia 1.9.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function multidimensional_function_space_operator end

function multidimensional_optimization_function end

function get_nsigma end