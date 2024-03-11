"""
FIXME Placeholder for a short summary about PosteriorAnalysis.
"""
module PosteriorAnalysis

using Compat: @compat
@compat public is_posterior, PosteriorArray, PosteriorVector, set_draw!, copy_draw,
   view_draw, each_index, map_posterior, collect_posterior, number_of_draws, each_draw,
   Elementwise, destructure_posterior

using ArgCheck: @argcheck
using Base: OneTo
import Base: ==, show, parent
using DocStringExtensions: SIGNATURES

####
#### generic code
####

"""
Supertype used for internal code organization only.
"""
abstract type Posterior{T} end

"""
$(SIGNATURES)

Test if a type is a posterior, ie whether it supports the interface defined in this
package, namely:

- [`number_of_draws`](@ref), [`each_draw`](@ref)
- [`copy_draw`](@ref), [`view_draw`](@ref), [`set_draw!`](@ref)
- [`map_posterior`](@ref)

See also [`each_index`](@ref), [`PosteriorArray`](@ref), [`PosteriorVector`](@ref),
[`collect_posterior`](@ref).
"""
@inline is_posterior(::Type) = false
@inline is_posterior(::Type{T}) where {T<:Posterior} = true
@inline is_posterior(x::T) where T = is_posterior(T)

"""
$(SIGNATURES)

Helper function to print posterior types. After printing a header, calls `f(io, p)` to
print type-specific information.
"""
function _print_posterior(f, io, p::Posterior{T}) where T
    N = number_of_draws(p)
    print(io, "« posterior draws: $N × $T")
    f(io, p)
    print(io, " »")
end

function show(io::IO, p::Posterior)
    _print_posterior((_...) -> nothing, io, p)
end

copy_draw(x, i) = x

view_draw(x, i) = copy_draw(x, i)

function _check_axis(axis, what)
    first(axis) == 1 || throw(ArgumentError(what * "needs to have 1-base indexing"))
end

####
#### vectors of arrays, compact representation
####

struct PosteriorArray{T,N,A<:AbstractArray} <: Posterior{Array{T,N}}
    posterior::A
    @doc """
    $(SIGNATURES)

    Make a collection of posterior draws from an array, where each draw is a slice
    `[:, :, …, :, i]` of the argument for all indices `i` on the last axis, which needs to
    be 1-based.

    Posterior draws of arrays of the same size and type, stored in a compact way with
    the last index for draws.

    `parent` can be used to access the underlying array. Specifically,

    ```julia
    parent(PosteriorArray(A)) ≡ A
    parent(posterior)[i..., j] == view_draw(posterior, j)[i...]
    ```

    Practically, it is usually an `Array`, but other types are accepted by the
    constructor.

    !!! note
        The type name itself and the constructor are part of the API, but internals
        and type parameters are not.
    """
    function PosteriorArray(posterior::A) where {T,M,A<:AbstractArray{T,M}}
        @argcheck M ≥ 1 "Not enough dimensions."
        _check_axis(last(axes(posterior)), "Last axis")
        new{T,M-1,A}(posterior)
    end
end

function show(io::IO, p::PosteriorArray{T}) where T
    _print_posterior(io, p) do io, p
        (; posterior) = p
        (a..., b) = axes(posterior)
        print(io, ", axes: ")
        join(io, [begin
                      c, d = extrema(a)
                      c == 1 ? string(d) : "($c:$d)"
                  end
                  for a in a],
             "×")
    end
end

number_of_draws(p::PosteriorArray) = last(size(p.posterior))

"""
$(SIGNATURES)

Helper function for generating index tuples of the form `(:, :, …, :, i)`. Internal.
"""
function _array_posterior_inds(p::PosteriorArray{T,N}, i) where {T,N}
    (ntuple(_ -> Colon(), Val(N))..., i)
end

copy_draw(p::PosteriorArray, i) = getindex(p.posterior, _array_posterior_inds(p, i)...)

view_draw(p::PosteriorArray, i) = view(p.posterior, _array_posterior_inds(p, i)...)

function set_draw!(p::PosteriorArray, d, i)
    setindex!(p.posterior, d, _array_posterior_inds(p, i)...)
end

function each_draw(p::PosteriorArray)
    (; posterior) = p
    eachslice(posterior; dims = ndims(posterior), drop = true)
end

"""
$(SIGNATURES)

Return an array-like view into posterior draws by indices of the draws.
"""
function each_index(p::PosteriorArray{T,N}) where {T,N}
    eachslice(p.posterior; dims = ntuple(identity, Val(N)), drop = true)
end

parent(p::PosteriorArray) = p.posterior

####
#### vectors without structure imposed
####

struct PosteriorVector{T,V<:AbstractVector{T}} <: Posterior{T}
    posterior::V
    @doc """
    $(SIGNATURES)

    Wrap a vector of posterior draws. The original vector can be accessed with `parent`.

    !!! note
        Whenever the draws are arrays of the same size, it is recommended to use
        [`PosteriorArray`](@ref). In this case, `PosteriorVector(v)` is equivalent to
        `PosteriorArray(stack(v))` for practical purposes.
    """
    function PosteriorVector(v::V) where {T,V <: AbstractVector{T}}
        _check_axis(axes(v, 1), "Vector")
        new{T,V}(v)
    end
end

number_of_draws(p::PosteriorVector) = length(p.posterior)

each_draw(p::PosteriorVector) = p.posterior

copy_draw(p::PosteriorVector, i) = p.posterior[i]

set_draw!(p::PosteriorVector, d, i) = p.posterior[i] = d

parent(p::PosteriorVector) = p.posterior

####
#### mapping
####

_common(f, x) = f(x)

"""
$(SIGNATURES)

Helper function to get the common `f` of something mapping from the `x`s.

That is, if all of `map(f, xs)` is the same (with `≡`, that is returned).

`nothing` is treated specially: it is ignored when comparing, and if all arguments yield
nothing, that is returned.

In case of a mismatch, throw an error. Internal.
"""
function _common(f, x, xs...)
    N = f(x)
    N2 = _common(f, xs...)
    if N ≡ N2 || N2 ≡ nothing
        N
    elseif N ≡ nothing
        N2
    else
        throw(ArgumentError("Mismatching $(f) ($N, $N2)"))
    end
end

"""
A placeholder container to initialize the collection of posterior values. Internal.
"""
struct _NoForm
    N::Int
end

number_of_draws(p::_NoForm) = p.N

"""
$(SIGNATURES)

Save the second argument in the first at index `i` if possible, returning a value that
is `≡p`.

If `p` cannot contain the second argument, widen and copy accordingly, save, and return
a value that is `≢p`.

Internal.
"""
function _save_or_widen!(p::_NoForm, r1::AbstractArray{T}, i) where T
    R = PosteriorArray(similar(Array{T}, (axes(r1)..., OneTo(number_of_draws(p)))))
    set_draw!(R, r1, i)
    R
end

function _save_or_widen!(p::_NoForm, r1::T, i) where T
    R = PosteriorVector(similar(Vector{T}, OneTo(number_of_draws(p))))
    set_draw!(R, r1, i)
    R
end

function _save_or_widen!(p::PosteriorArray{T,N}, r::AbstractArray{T,N}, i) where {T,N}
    set_draw!(p, r, i)
    p
end

function _save_or_widen!(p::PosteriorVector{T}, r::T, i) where T
    set_draw!(p, r, i)
    p
end

function _save_or_widen!(p::Posterior{T1}, r::T2, i) where {T1,T2} # fallback
    T = Base.promote_typejoin(T1, T2)
    v = Vector{T}(undef, number_of_draws(p))
    for j in 1:(i-1)
        v[j] = copy_draw(p, j)
    end
    v[i] = r
    PosteriorVector(v)
end

function _map_posterior!(p, i, f, args...)
    N = number_of_draws(p)
    for j in i:N
        draws = map(p -> view_draw(p, j), args)
        r = f(draws...)
        p′ = _save_or_widen!(p, r, j)
        if p ≢ p′
            return _map_posterior!(p′, j + 1, f, args...)
        end
    end
    p
end

"""
$(SIGNATURES)

The `number_of_draws(argument)` for posteriors, `nothing` for all other types. Internal
helper function.
"""
_number_of_draws_or_nothing(p::Posterior) = number_of_draws(p)
_number_of_draws_or_nothing(_) = nothing

"""
$(SIGNATURES)

Map posterior draws using `f`.

If an argument is not a posterior, it is applied as is for each call. Whenever possible,
collect the result in a compact form (eg [`PosteriorArray`](@ref)).

If no arguments are posteriors, `f(args...)` is returned, otherwise the result is a
posterior.
"""
@noinline function map_posterior(f, args...)
    N = _common(_number_of_draws_or_nothing, args...)
    N ≡ nothing && return f(args...)
    @argcheck N ≥ 1
    _map_posterior!(_NoForm(N), 1, f, args...)
end

function collect_posterior(itr)
    _collect_posterior(Base.IteratorSize(itr), itr)
end

function _not_vec_error()
    throw(ArgumentError("Can only accept finite, vector-like iterators as a posterior."))
end

function _collect_posterior(::Union{Base.HasLength,Base.SizeUnknown,Base.HasShape{1}}, itr)
    v = collect(itr)
    v isa Vector || _not_vec_error()
    p = PosteriorVector(v)
    if view_draw(p, 1) isa AbstractArray
        # NOTE collecting twice is inefficient; didn't bother to write code for that case
        map_posterior(identity, p)
    else
        p
    end
end

_collect_posterior(::Union{Base.HasShape,Base.IsInfinite}, itr) = _not_vec_error()

"""
$(SIGNATURES)

A wrapper equivalent to `(x...)->f(x...)`, which results in direct application in
`map_posterior` when possible.
"""
struct Elementwise{F}
    f::F
end

(f::Elementwise)(xs...) = map(f.f, xs...)

function map_posterior(f::Elementwise, args::PosteriorArray...)
    posteriors = map(x -> x.posterior, args)
    @argcheck _common(size, posteriors...) ≢ nothing
    PosteriorArray(map(f, posteriors...))
end

map_posterior(f::Elementwise, args...) = map_posterior((x...,) -> f.f.(x...), args...)

# FIXME implement hashing, add tests for == too
(==)(a::PosteriorArray, b::PosteriorArray) = a.posterior == b.posterior
(==)(a::PosteriorVector, b::PosteriorVector) = a.posterior == b.posterior
(==)(a::Posterior, b::Posterior) = each_draw(a) == each_draw(b)

"""
$(SIGNATURES)

Destructure a `PosteriorArray` of `Tuple`s or `NamedTuple`s elementwise.
"""
function destructure_posterior(p::PosteriorArray{T}) where {N, T <: NTuple{N}}
    (; posterior) = p
    ntuple(i -> PosteriorArray(map(x -> x[i], posterior)), Val(N))
end

function destructure_posterior(p::PosteriorArray{T}) where {K, T <: NamedTuple{K}}
    (; posterior) = p
    NamedTuple{K}(ntuple(i -> PosteriorArray(map(x -> x[i], posterior)), Val(fieldcount(T))))
end

end # module
