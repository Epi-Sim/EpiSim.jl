using Pkg
episim_src_dir = @__DIR__
episim_base_dir = dirname(episim_src_dir)
Pkg.activate(episim_base_dir)
# Pkg.instantiate()

using EpiSim
using JSON
using ArgParse
using Logging

function parse_batch_command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--batch-folder", "-b"
            help = "Parent folder containing multiple run subdirectories"
            required = true
        "--data-folder", "-d"
            help = "Folder containing static model data (metapopulation, etc.)"
            required = true
        "--workers", "-w"
            help = "Number of threads to use (default: auto)"
            default = "auto"
    end
    return parse_args(s)
end

function run_batch_simulation(config_path, data_folder, instance_folder)
    try
        # Load Config
        config = JSON.parsefile(config_path)
        
        # Determine Engine
        # We assume MMCACovid19 as per current synthetic setup, 
        # or verify if we need to support Vac via config inspection
        engine_name = config["simulation"]["engine"]
        engine = EpiSim.get_engine(engine_name)
        
        # Validate Config
        engine = EpiSim.validate_config(config)
        
        # Run Simulation
        # run_engine_io handles reading inputs, init variables, and running the loop
        EpiSim.run_engine_io(engine, config, data_folder, instance_folder)
        
        # println("Completed: $(basename(instance_folder))")
        
    catch e
        @error "Failed run: $(basename(instance_folder))" exception=(e, catch_backtrace())
    end
end

function main()
    args = parse_batch_command_line()
    batch_folder = args["batch-folder"]
    data_folder = args["data-folder"]
    
    # Find all run folders
    # We look for config_auto_py.json files to identify valid run directories
    configs = []
    for (root, dirs, files) in walkdir(batch_folder)
        if "config_auto_py.json" in files
            push!(configs, joinpath(root, "config_auto_py.json"))
        end
    end
    
    println("Found $(length(configs)) simulations to run.")
    
    # Threaded Execution
    # Julia starts with -t/--threads threads. 
    # @threads distributes the loop iterations across these threads.
    Threads.@threads for config_path in configs
        instance_folder = dirname(config_path)
        # println("Starting on thread $(Threads.threadid()): $(basename(instance_folder))")
        run_batch_simulation(config_path, data_folder, instance_folder)
    end
    
    println("Batch execution complete.")
end

main()