"""
FIXME Placeholder for a short summary about PosteriorAnalysis.
"""
module PosteriorAnalysis

# NOTE we don't export anything, use public when Julia 1.11 comes out
# public number_of_draws, copy_draw, view_draw, set_draw!, each_index
#     posterior_last_axis, posterior_vector, map_posterior, collect_posterior

using ArgCheck: @argcheck
using Base: OneTo
using DocStringExtensions: SIGNATURES

####
#### generic code
####

abstract type Posterior{T} end

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

function Base.show(io::IO, p::Posterior)
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
    function PosteriorArray(posterior::A) where {T,M,A<:AbstractArray{T,M}}
        @argcheck M ≥ 1 "Not enough dimensions."
        _check_axis(last(axes(posterior)), "Last axis")
        new{T,M-1,A}(posterior)
    end
end

"""
$(SIGNATURES)

Make a collection of posterior draws from an array, where each draw is a slice `[:, :,
…, :, i]` of the argument for all indices `i` on the last axis, which needs to be
1-based.
"""
posterior_last_axis(a::AbstractArray) = PosteriorArray(a)

function Base.show(io::IO, p::PosteriorArray{T}) where T
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

####
#### vectors without structure imposed
####

struct PosteriorVector{T,V<:AbstractVector{T}} <: Posterior{T}
    posterior::V
    function PosteriorVector(v::V) where {T,V <: AbstractVector{T}}
        _check_axis(axes(v, 1), "Vector")
        new{T,V}(v)
    end
end

posterior_vector(v::AbstractVector) = PosteriorVector(v)

number_of_draws(p::PosteriorVector) = length(p.posterior)

copy_draw(p::PosteriorVector, i) = p.posterior[i]

set_draw!(p::PosteriorVector, d, i) = p.posterior[i] = d

####
#### mapping
####

_common_number_of_draws(p::Posterior) = number_of_draws(p)

_common_number_of_draws(x) = nothing

_common_number_of_draws(x1, xs...) = _common_number_of_draws(xs...)

"""
$(SIGNATURES)

Helper function to get the common number of draws, or `nothing` if none of the arguments
is a posterior. In case of a mismatch, throw an error. Internal.
"""
function _common_number_of_draws(p::Posterior, xs...)
    N = number_of_draws(p)
    N2 = _common_number_of_draws(xs...)
    if N2 ≡ nothing || N == N2
        N
    else
        throw(ArgumentError("Mismatching number of draws ($N, $N2)"))
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
        r = f(map(p -> view_draw(p, j), args)...)
        p′ = _save_or_widen!(p, r, j)
        if p ≢ p′
            return _map_posterior!(p′, j + 1, f, args...)
        end
    end
    p
end

"""
$(SIGNATURES)

Map posterior draws using `f`.

If an argument is not a posterior, it is applied as is for each call.

If no arguments are posteriors, `f(args...)` is returned, otherwise the result is a
posterior.
"""
function map_posterior(f, args...)
    N = _common_number_of_draws(args...)
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
    p = posterior_vector(v)
    if view_draw(p, 1) isa AbstractArray
        # NOTE collecting twice is inefficient; didn't bother to write code for that case
        map_posterior(identity, p)
    else
        p
    end
end

_collect_posterior(::Union{Base.HasShape,Base.IsInfinite}, itr) = _not_vec_error()

end # module
