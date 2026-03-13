using Pkg
episim_src_dir = @__DIR__
episim_base_dir = dirname(episim_src_dir)
Pkg.activate(episim_base_dir)

using Distributed
import Distributed: remotecall_eval
using EpiSim
using JSON
using ArgParse
using Logging
using Dates

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
            help = "Number of parallel workers (0 or 1 = sequential, N = distributed with N workers)"
            default = "auto"
        "--worker-timeout"
            help = "Timeout in seconds for each simulation (0 = no timeout)"
            default = "3600"
    end
    return parse_args(s)
end

const WORKER_CODE = quote
    function write_error_file(instance_folder, e, bt)
        error_data = Dict(
            "timestamp" => string(now()),
            "run_folder" => basename(instance_folder),
            "error_type" => string(typeof(e)),
            "error_message" => string(e),
            "stacktrace" => sprint(show, bt)
        )
        error_path = joinpath(instance_folder, "ERROR.json")
        open(error_path, "w") do f
            JSON.print(f, error_data, 2)
        end
    end

    function run_batch_simulation(config_path, data_folder, instance_folder)
        try
            config = JSON.parsefile(config_path)
            engine_name = config["simulation"]["engine"]
            engine = EpiSim.get_engine(engine_name)
            
            output_file = joinpath(instance_folder, "output", "observables.nc")
            if isfile(output_file)
                return (success = true, config_path = config_path, instance_folder = instance_folder, skipped = true)
            end
            
            engine = EpiSim.validate_config(config)
            EpiSim.run_engine_io(engine, config, data_folder, instance_folder)
            
            return (success = true, config_path = config_path, instance_folder = instance_folder, skipped = false)
            
        catch e
            bt = catch_backtrace()
            @error "Failed run: $(basename(instance_folder))" exception = (e, bt)
            write_error_file(instance_folder, e, bt)
            return (success = false, config_path = config_path, instance_folder = instance_folder, skipped = false, error = string(e))
        end
    end
end

eval(WORKER_CODE)

function find_configs(batch_folder)
    configs = String[]
    for (root, dirs, files) in walkdir(batch_folder)
        if "config_auto_py.json" in files
            push!(configs, joinpath(root, "config_auto_py.json"))
        end
    end
    return configs
end

function prepare_directories(configs)
    for config_path in configs
        instance_folder = dirname(config_path)
        output_path = joinpath(instance_folder, "output")
        if !isdir(output_path)
            mkpath(output_path)
        end
    end
end

function run_sequential(configs, data_folder)
    @info "Running $(length(configs)) simulations sequentially..."
    
    results = []
    for config_path in configs
        result = run_batch_simulation(config_path, data_folder, dirname(config_path))
        push!(results, result)
    end
    
    return results
end



function load_packages_on_workers(worker_pids)
    @sync for pid in worker_pids
        @async remotecall_eval(Main, pid, quote
            using EpiSim
            using JSON
            using Dates
            using Logging
        end)
    end
end

function load_worker_function_on_workers(worker_pids)
    @sync for pid in worker_pids
        @async remotecall_eval(Main, pid, WORKER_CODE)
    end
end

function run_distributed(configs, data_folder, n_workers, worker_timeout)
    @info "Running $(length(configs)) simulations with $n_workers distributed workers..."
    @info "Each worker uses single-threaded mode (-t 1) for HDF5 safety"
    
    @info "Starting parallel simulation execution..."
    
    results = pmap(configs; 
                   retry_delays = ExponentialBackOff(n=2, first_delay=1.0),
                   on_error = e -> begin
                       @error "Simulation failed with exception" exception=e
                       (success = false, config_path = "", instance_folder = "", skipped = false, error = string(e))
                   end) do config_path
        run_batch_simulation(config_path, data_folder, dirname(config_path))
    end
    
    return results
end

function write_results_summary(batch_folder, results)
    succeeded = count(r -> r.success && !r.skipped, results)
    skipped = count(r -> r.skipped, results)
    failed = count(r -> !r.success, results)
    
    results_dict = Dict(
        "total" => length(results),
        "succeeded" => succeeded,
        "skipped" => skipped,
        "failed" => failed,
        "failures" => [basename(r.instance_folder) for r in results if !r.success]
    )
    
    results_path = joinpath(batch_folder, "BATCH_RESULTS.json")
    open(results_path, "w") do f
        JSON.print(f, results_dict, 2)
    end
    
    println("\nBatch execution complete.")
    println("  Total: $(length(results))")
    println("  Succeeded: $succeeded")
    println("  Skipped (already complete): $skipped")
    println("  Failed: $failed")
    
    return failed == 0
end

function main()
    args = parse_batch_command_line()
    batch_folder = args["batch-folder"]
    data_folder = args["data-folder"]
    workers_arg = args["workers"]
    worker_timeout = parse(Int, args["worker-timeout"])
    
    println("=" ^ 60)
    println("EpiSim Batch Runner")
    println("=" ^ 60)
    println("Batch folder: $batch_folder")
    println("Data folder: $data_folder")
    
    configs = find_configs(batch_folder)
    
    if isempty(configs)
        @error "No config_auto_py.json files found in $batch_folder"
        exit(1)
    end
    
    println("Found $(length(configs)) simulations to run.")
    
    n_configs = length(configs)
    
    if workers_arg == "auto"
        n_workers = min(Sys.CPU_THREADS, n_configs)
        n_workers = max(1, n_workers)
    else
        n_workers = parse(Int, workers_arg)
    end
    
    println("Pre-creating output directories...")
    prepare_directories(configs)
    
    results = nothing
    worker_pids = Int[]
    
    try
        if n_workers <= 1
            println("Mode: Sequential execution (workers=$n_workers)")
            results = run_sequential(configs, data_folder)
        else
            println("Mode: Distributed execution with $n_workers workers")
            println("Note: Each worker runs single-threaded (-t 1) for HDF5 safety")
            
            actual_workers = min(n_workers, n_configs)
            
            exeflags = ["--project=$episim_base_dir", "-t", "1"]
            
            println("Spawning $actual_workers worker processes...")
            worker_pids = addprocs(actual_workers; exeflags=exeflags)
            
            println("Loading packages on all workers...")
            load_packages_on_workers(worker_pids)
            
            println("Loading worker function on all workers...")
            load_worker_function_on_workers(worker_pids)
            
            println("Packages loaded on all workers")
            
            results = run_distributed(configs, data_folder, actual_workers, worker_timeout)
        end
        
        success = write_results_summary(batch_folder, results)
        
        if !success
            exit(1)
        end
        
    finally
        if !isempty(worker_pids)
            println("Cleaning up worker processes...")
            rmprocs(worker_pids)
        end
    end
end

if myid() == 1 && abspath(PROGRAM_FILE) == @__FILE__
    main()
end
