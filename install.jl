using Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()

using ArgParse
using PackageCompiler

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--compile", "-c"
            help = "Compile the simulator into a single precompiled excecutable"
            action = :store_true
        "--incremental", "-i"
            help = "Compile the simulator incrementally. NOT IN HPC!"
            action = :store_true
            default = false
        "--target", "-t"
            help = "Target folder where the single excecutable will be stored"
            default ="."
        "--update", "-u"
            help = "Update dependencies"
            action = :store_true
            default = false
    end
    return parse_args(s)
end



args = parse_commandline()
@assert isdir(args["target"]) "Target folder $(args["target"]) does not exist"

if args["update"]
    Pkg.update()
    Pkg.instantiate()
    Pkg.precompile()
end

episim_path = joinpath(args["target"], "episim")

if args["compile"]
    build_folder = "build"
    println("Attempting to compile the application...")
    try
        create_app(pwd(), build_folder,
            force=true,
            incremental=args["incremental"],
            include_transitive_dependencies=true,
            filter_stdlibs=false,
            precompile_execution_file=["src/EpiSim.jl"])

        bin_path = abspath(joinpath(build_folder, "bin", "EpiSim"))

        if islink(episim_path) || isfile(episim_path)
            rm(episim_path, force=true)
        end
        symlink(bin_path, episim_path)
        println("Compilation successful. 'episim' command created at $(episim_path)")
    catch e
        println("Compilation failed: $e")
        rethrow(e)
    end
else
    println("Creating a wrapper script for 'episim' as compilation was not requested.")
    
    project_dir = abspath(@__DIR__)
    run_script_path = joinpath(project_dir, "src", "run.jl")

    if islink(episim_path) || isfile(episim_path)
        rm(episim_path, force=true)
    end

    open(episim_path, "w") do f
        println(f, "#!/bin/sh")
        println(f, "exec julia --project=\"$project_dir\" \"$run_script_path\" \"\$@\"")
    end
    chmod(episim_path, 0o755)
    println("'episim' wrapper script created at $(episim_path)")
end