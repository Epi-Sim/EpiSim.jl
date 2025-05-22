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

if args["compile"]
    build_folder = "build"
    create_app(pwd(), build_folder, 
        force=true, 
        incremental=args["incremental"],
        include_transitive_dependencies=true,
        filter_stdlibs=false,
        precompile_execution_file=["src/EpiSim.jl"])
    bin_path = abspath(joinpath(build_folder, "bin", "EpiSim"))
    symlink_path = joinpath(args["target"], "episim")
    if !islink(symlink_path)
        symlink(bin_path, symlink_path)
    end
end
