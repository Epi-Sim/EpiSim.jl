using Pkg
episim_src_dir = @__DIR__
episim_base_dir = dirname(episim_src_dir)
Pkg.activate(episim_base_dir)
# Pkg.instantiate()

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
            help = "Number of threads to use (default: auto)"
            default = "auto"
    end
    return parse_args(s)
end

function write_error_file(instance_folder, e, bt)
    """Write error details to ERROR.json in the failed run directory."""
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
        # Load Config
        config = JSON.parsefile(config_path)

        # Determine Engine
        # We assume MMCACovid19 as per current synthetic setup,
        # or verify if we need to support Vac via config inspection
        engine_name = config["simulation"]["engine"]
        engine = EpiSim.get_engine(engine_name)

        # Check if already completed (skip if observables.nc exists)
        # This allows incremental batch execution without re-running previous batches
        output_file = joinpath(instance_folder, "output", "observables.nc")
        if isfile(output_file)
             # Verify it's not empty/corrupt? For now assume existence means success.
             # println("Skipping already completed: $(basename(instance_folder))")
             return true
        end

        # Validate Config
        engine = EpiSim.validate_config(config)

        # Run Simulation
        # run_engine_io handles reading inputs, init variables, and running the loop
        EpiSim.run_engine_io(engine, config, data_folder, instance_folder)

        # println("Completed: $(basename(instance_folder))")

    catch e
        bt = catch_backtrace()
        @error "Failed run: $(basename(instance_folder))" exception=(e, bt)
        write_error_file(instance_folder, e, bt)
        return false  # Indicate failure
    end
    return true  # Indicate success
end

function main_with_tracking()
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

    # Track results
    results = Dict(
        "total" => length(configs),
        "succeeded" => 0,
        "failed" => 0,
        "failures" => []
    )

    # Threaded Execution
    # Julia starts with -t/--threads threads.
    # @threads distributes the loop iterations across these threads.
    Threads.@threads for config_path in configs
        instance_folder = dirname(config_path)
        # println("Starting on thread $(Threads.threadid()): $(basename(instance_folder))")
        success = run_batch_simulation(config_path, data_folder, instance_folder)
        if success
            results["succeeded"] += 1
        else
            results["failed"] += 1
            push!(results["failures"], basename(instance_folder))
        end
    end

    # Write results summary
    results_path = joinpath(batch_folder, "BATCH_RESULTS.json")
    open(results_path, "w") do f
        JSON.print(f, results, 2)
    end

    println("Batch execution complete.")
    println("  Succeeded: $(results["succeeded"])")
    println("  Failed: $(results["failed"])")

    # Exit with error if any failures
    if results["failed"] > 0
        exit(1)
    end
end

main_with_tracking()