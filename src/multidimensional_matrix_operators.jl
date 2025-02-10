"""
    MultidimensionalMatrixOperator{Dim, T}

Multidimensional operator that represents a first-derivative operator based on a matrix.

To be obtain the derivative operator in a specific direction, use `Matrix(D, dim)`. The boundary operator in
a specific direction can be obtained with `mass_matrix_boundary(D, dim)`.

See also [`multidimensional_function_space_operator`](@ref), [`GlaubitzIskeLampertÖffner2024`](@ref).
"""
@auto_hash_equals struct MultidimensionalMatrixOperator{Dim,T,NodesType,DType<:AbstractMatrix{T},SourceOfCoefficients} <: AbstractNonperiodicDerivativeOperator{T}
    grid::NodesType # length(grid) == N, e.g. Vector{SVector{Dim, T}} or `NodeSet` from KernelInterpolation.jl
    on_boundary::Vector{Bool} # length(on_boundary) == N
    normals::Vector{SVector{Dim,T}} # length(normals) == N_boundary < N
    weights::Vector{T} # length(weights) == N
    weights_boundary::Vector{T} # length(weights_boundary) == N_boundary < N
    Ds::NTuple{Dim,DType}
    accuracy_order::Int
    source::SourceOfCoefficients

    function MultidimensionalMatrixOperator(nodes::NodesType,
        on_boundary::Vector{Bool},
        normals::Vector{SVector{Dim,T}},
        weights::Vector{T}, weights_boundary::Vector{T},
        Ds::NTuple{Dim,DType}, accuracy_order::Int,
        source::SourceOfCoefficients) where {Dim,T<:Real,NodesType,DType<:AbstractMatrix{T},SourceOfCoefficients}
        new{Dim,T,NodesType,DType,SourceOfCoefficients}(nodes, on_boundary, normals, weights, weights_boundary, Ds, accuracy_order, source)
    end
end

Base.ndims(::MultidimensionalMatrixOperator{Dim}) where {Dim} = Dim
derivative_order(::MultidimensionalMatrixOperator) = 1
Matrix(D::MultidimensionalMatrixOperator, dim::Int) = Matrix(D.Ds[dim])

source_of_coefficients(D::MultidimensionalMatrixOperator) = D.source

function integrate(func, u, D::MultidimensionalMatrixOperator)
    return integrate(func, u, D.weights)
end

function integrate_boundary(func, u, D::MultidimensionalMatrixOperator, dim)
    return integrate(func, u, weights_boundary_scaled(D, dim))
end

mass_matrix(D::MultidimensionalMatrixOperator) = Diagonal(D.weights)
# TODO: more efficient
weights_boundary(D::MultidimensionalMatrixOperator) = get_weight_boundary.(Ref(D), 1:length(grid(D)))
weights_boundary_scaled(D::MultidimensionalMatrixOperator, dim::Int) = get_weight_boundary_scaled.(Ref(D), Ref(dim), 1:length(grid(D)))
mass_matrix_boundary(D::MultidimensionalMatrixOperator, dim::Int) = Diagonal(weights_boundary_scaled(D, dim))

Base.eltype(::MultidimensionalMatrixOperator{Dim,T}) where {Dim,T} = T

function scale_by_mass_matrix!(u::AbstractVector, D::MultidimensionalMatrixOperator, factor=true)
    N, _ = size(D)
    @boundscheck begin
        @argcheck N == length(u)
    end

    @inbounds @simd for i in eachindex(u, D.weights)
        u[i] = factor * u[i] * D.weights[i]
    end

    return u
end

function scale_by_inverse_mass_matrix!(u::AbstractVector, D::MultidimensionalMatrixOperator, factor=true)
    N, _ = size(D)
    @boundscheck begin
        @argcheck N == length(u)
    end

    @inbounds @simd for i in eachindex(u, D.weights)
        u[i] = factor * u[i] / D.weights[i]
    end

    u
end

function get_weight(D::MultidimensionalMatrixOperator, i::Int)
    @unpack weights = D
    N, _ = size(D)
    @boundscheck begin
        @argcheck 1 <= i <= N
    end
    @inbounds ω = weights[i]
    ω
end

function get_weight_boundary(D::MultidimensionalMatrixOperator, i::Int)
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

function get_weight_boundary_scaled(D::MultidimensionalMatrixOperator, dim::Int, i::Int)
    @unpack normals, on_boundary = D
    if !on_boundary[i]
        return zero(eltype(D))
    end
    ω = get_weight_boundary(D, i)
    j = sum(view(on_boundary, 1:i))
    ω * normals[j][dim]
end

function Base.show(io::IO, D::MultidimensionalMatrixOperator)
    if get(io, :compact, false)
        summary(io, D)
    else
        x = grid(D)
        print(io, ndims(D), "-dimensional function space first-derivative operator {T=", eltype(D), "}")
        print(io, " on ", length(x), " nodes")
    end
end

# TODO: mul! How? Depends on direction

function lower_bandwidth(D::MultidimensionalMatrixOperator)
    size(D, 1) - 1
end

function upper_bandwidth(D::MultidimensionalMatrixOperator)
    size(D, 1) - 1
end

function accuracy_order(D::MultidimensionalMatrixOperator)
    D.accuracy_order
end

# TODO
function left_boundary_weight(D::MultidimensionalMatrixOperator)
    @inbounds retval = D.weights[begin]
    retval
end

function right_boundary_weight(D::MultidimensionalMatrixOperator)
    @inbounds retval = D.weights[end]
    retval
end
