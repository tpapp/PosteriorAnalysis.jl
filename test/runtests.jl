using PosteriorAnalysis
using Test

@testset "posterior vector basic operations" begin
    N = 10
    p = posterior_vector(fill(NaN, N))
    for i in 1:N
        w = 1.0 .+ i
        set_draw!(p, w, i)
        @test copy_draw(p, i) == view_draw(p, i) == w
    end
end

@testset "posterior array basic operations" begin
    N = 10
    M = 5
    p = posterior_last_axis(fill(NaN, M, N))
    v = Float64.(1:M)
    for i in 1:N
        w = v .+ i
        set_draw!(p, w, i)
        @test copy_draw(p, i) == view_draw(p, i) == w
    end
end

@testset "map_posterior (to array)" begin
    N = 4
    A = randn(3, N)
    B = randn(N)
    p1 = posterior_last_axis(A)
    p2 = posterior_vector(B)
    p12 = map_posterior((x, y) -> x .+ y, p1, p2)
    @test number_of_draws(p12) == N
    @test p12 isa PosteriorAnalysis.PosteriorArray
    for i in 1:N
        @test copy_draw(p1, i) .+ copy_draw(p2, i) == copy_draw(p12, i)
    end
end

@testset "map_posterior (widening)" begin
    N = 4
    A = rand(1:5, 3, N)
    B = Union{Int,Float64}[isodd(i) ? 1 : 2.0 for i in 1:N]
    p1 = posterior_last_axis(A)
    p2 = posterior_vector(B)
    p12 = map_posterior((x, y) -> x .+ y, p1, p2)
    @test number_of_draws(p12) == N
    @test p12 isa PosteriorAnalysis.PosteriorVector
    for i in 1:N
        @test copy_draw(p1, i) .+ copy_draw(p2, i) == copy_draw(p12, i)
    end
end

@testset "collect posterior" begin
    N = 10

    f1(i) = (1:3) .* i
    p1 = collect_posterior((f1(i) for i in 1:N))
    @test p1 isa PosteriorAnalysis.PosteriorArray
    for i in 1:N
        @test view_draw(p1, i) == f1(i)
    end

    f2(i) = isodd(i) ? i : f1(i)
    p2 = collect_posterior((f2(i) for i in 1:N))
    @test p2 isa PosteriorAnalysis.PosteriorVector
    for i in 1:N
        @test view_draw(p2, i) == f2(i)
    end
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
