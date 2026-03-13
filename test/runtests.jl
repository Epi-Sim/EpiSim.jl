using Test
using NCDatasets
using EpiSim

episim_src_dir = @__DIR__
episim_base_dir = dirname(episim_src_dir)
model_dir = joinpath(episim_base_dir, "models", "mitma")

const T_STEPS = 20
const G_AGENTS = 3
const M_METAPOPS = 2850

function verify_observables_nc(filepath)
    @test isfile(filepath)
    
    ds = NCDataset(filepath)
    
    try
        @testset "Required dimensions" begin
            @test haskey(ds.dim, "T")
            @test haskey(ds.dim, "G")
            @test haskey(ds.dim, "M")
        end
        
        @testset "Required variables" begin
            required_vars = ["new_infected", "new_hospitalized", "new_deaths"]
            for var in required_vars
                @test haskey(ds, var)
            end
        end
        
        @testset "Dimension sizes" begin
            if haskey(ds.dim, "T")
                @test ds.dim["T"] == T_STEPS
            end
            if haskey(ds.dim, "G")
                @test ds.dim["G"] == G_AGENTS
            end
            if haskey(ds.dim, "M")
                @test ds.dim["M"] == M_METAPOPS
            end
        end
        
        @testset "Data validity" begin
            if haskey(ds, "new_infected")
                infected = ds["new_infected"][:]
                @test !all(isnan.(infected))
                @test all(infected .>= 0)
            end
        end
        
    finally
        close(ds)
    end
end

@testset "EpiSim Integration Tests" begin
    
    @testset "MMCACovid19 engine" begin
        output_dir = mktempdir(prefix="episim_test_mmcacovid19_")
        
        try
            args = Dict(
                "config" => joinpath(model_dir, "config_MMCACovid19.json"),
                "data-folder" => model_dir,
                "instance-folder" => output_dir,
                "initial-condition" => "",
                "start-date" => "2020-02-09",
                "end-date" => "2020-02-28",
                "export-compartments-time-t" => nothing,
                "export-compartments-full" => nothing
            )
            
            EpiSim.execute_run(args)
            
            observables_path = joinpath(output_dir, "output", "observables.nc")
            verify_observables_nc(observables_path)
            
        finally
            rm(output_dir; recursive=true, force=true)
        end
    end
    
    @testset "MMCACovid19Vac engine" begin
        output_dir = mktempdir(prefix="episim_test_mmcacovid19vac_")
        
        try
            args = Dict(
                "config" => joinpath(model_dir, "config_MMCACovid19-vac.json"),
                "data-folder" => model_dir,
                "instance-folder" => output_dir,
                "initial-condition" => "",
                "start-date" => "2020-02-09",
                "end-date" => "2020-02-28",
                "export-compartments-time-t" => nothing,
                "export-compartments-full" => nothing
            )
            
            EpiSim.execute_run(args)
            
            observables_path = joinpath(output_dir, "output", "observables.nc")
            verify_observables_nc(observables_path)
            
        finally
            rm(output_dir; recursive=true, force=true)
        end
    end
    
    @testset "Error handling" begin
        @testset "Missing config file" begin
            args = Dict(
                "config" => joinpath(model_dir, "nonexistent_config.json"),
                "data-folder" => model_dir,
                "instance-folder" => mktempdir(),
                "initial-condition" => "",
                "start-date" => nothing,
                "end-date" => nothing,
                "export-compartments-time-t" => nothing,
                "export-compartments-full" => nothing
            )
            
            @test_throws Exception EpiSim.execute_run(args)
        end
    end
    
end
