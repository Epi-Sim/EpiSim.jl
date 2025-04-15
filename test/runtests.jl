using Test
using DataFrames
using EpiSim




command = "run"

episim_src_dir = @__DIR__
episim_base_dir = dirname(episim_src_dir)

model_dir   = joinpath(episim_base_dir, "models", "mitma")
run_folder  = episim_base_dir


args = Dict([
    "config" => nothing,
    "data-folder" => model_dir,
    "instance-folder" => run_folder,
    "initial-condition" => "",
    "start-date" => nothing,
    "end-date" => nothing,
    "export-compartments-time-t" => nothing,
    "export-compartments-full" => nothing
    ])


@testset "Testing Episim command: run (engine::MMCACovid19)" begin
    config_json = "config_MMCACovid19.json"
    args["config"] =joinpath(model_dir, config_json)
    EpiSim.execute_run(args)
end

@testset "Testing Episim command: run (engine::MMCACovid19-vac)" begin
    config_json = "config_MMCACovid19-vac.json"
    args["config"] =joinpath(model_dir, config_json)
    EpiSim.execute_run(args)
end