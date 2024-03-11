using PosteriorAnalysis: PosteriorArray, PosteriorVector, set_draw!, copy_draw,
    view_draw, each_index, map_posterior, collect_posterior, number_of_draws, each_draw,
    is_posterior, Elementwise

using PosteriorAnalysis
using Test

@testset "posterior vector basic operations" begin
    N = 10
    p = PosteriorVector(fill(NaN, N))
    for i in 1:N
        w = 1.0 .+ i
        set_draw!(p, w, i)
        @test copy_draw(p, i) == view_draw(p, i) == w
    end
end

@testset "posterior array basic operations" begin
    N = 10
    M = 5
    p = PosteriorArray(fill(NaN, M, N))
    v = Float64.(1:M)
    for i in 1:N
        w = v .+ i
        set_draw!(p, w, i)
        @test copy_draw(p, i) == view_draw(p, i) == w
    end

    s = each_index(p)
    @test size(s) == (M, )
    @test eltype(s) <: AbstractVector{Float64}
    for j in 1:M
        @test s[j] == [view_draw(p, i)[j] for i in 1:N]
    end
end

@testset "map_posterior (to array)" begin
    N = 4
    A = randn(3, N)
    B = randn(N)
    p1 = PosteriorArray(A)
    p2 = PosteriorVector(B)
    p12 = map_posterior((x, y) -> x .+ y, p1, p2)
    @test number_of_draws(p12) == N
    @test p12 isa PosteriorArray
    for i in 1:N
        @test copy_draw(p1, i) .+ copy_draw(p2, i) == copy_draw(p12, i)
    end
end

@testset "map_posterior (widening)" begin
    N = 4
    A = rand(1:5, 3, N)
    B = Union{Int,Float64}[isodd(i) ? 1 : 2.0 for i in 1:N]
    p1 = PosteriorArray(A)
    p2 = PosteriorVector(B)
    p12 = map_posterior((x, y) -> x .+ y, p1, p2)
    @test number_of_draws(p12) == N
    @test p12 isa PosteriorVector
    for i in 1:N
        @test copy_draw(p1, i) .+ copy_draw(p2, i) == copy_draw(p12, i)
    end
end

@testset "collect posterior" begin
    N = 10

    f1(i) = (1:3) .* i
    p1 = collect_posterior((f1(i) for i in 1:N))
    @test p1 isa PosteriorArray
    for i in 1:N
        @test view_draw(p1, i) == f1(i)
    end

    f2(i) = isodd(i) ? i : f1(i)
    p2 = collect_posterior((f2(i) for i in 1:N))
    @test p2 isa PosteriorVector
    for i in 1:N
        @test view_draw(p2, i) == f2(i)
    end
end

@testset "each_draw, is_posterior, parent, and stacking" begin
    A = randn(4, 5, 6)
    pA = PosteriorArray(A)
    @test is_posterior(pA)
    @test parent(pA) ≡ A

    V = collect(eachslice(A; dims = 3))
    @test each_draw(pA) == V
    pV = PosteriorVector(V)
    @test is_posterior(pV)
    @test parent(pV) ≡ V
    @test parent(PosteriorArray(stack(V))) == A
    @test each_draw(pV) == V

    @test !is_posterior("a fish")
end

@testset "elementwise" begin
    A = randn(5, 4)
    B = randn(5, 4)
    a = PosteriorArray(A)
    b = PosteriorArray(B)
    C = map_posterior(Elementwise(+), a, b)
    C2 = map_posterior(Elementwise(+), a, PosteriorVector(eachcol(B)))
    c = map_posterior(Elementwise(+), A, B)
    @test c == A .+ B
    @test parent(C) == c
    @test C == C2
end

using JET
@testset "static analysis with JET.jl" begin
    @test isempty(JET.get_reports(report_package(PosteriorAnalysis, target_modules=(PosteriorAnalysis,))))
end

@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(PosteriorAnalysis; ambiguities = false)
    # testing separately, cf https://github.com/JuliaTesting/Aqua.jl/issues/77
    Aqua.test_ambiguities(PosteriorAnalysis)
end
