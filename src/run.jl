using Pkg

episim_src_dir = @__DIR__
episim_base_dir = dirname(episim_src_dir)
Pkg.activate(episim_base_dir)

if get(ENV, "EPISIM_INSTANTIATE_ON_STARTUP", "1") != "0"
    Pkg.instantiate()
end

using EpiSim

EpiSim.main()
