include("common.jl")

function update_config!(config, cmd_line_args)
    # Define dictionary containing epidemic parameters

    # overwrite config with command line
    if cmd_line_args["start-date"] !== nothing
        config["simulation"]["start_date"] = cmd_line_args["start-date"]
    end
    if cmd_line_args["end-date"] !== nothing
        config["simulation"]["end_date"] = cmd_line_args["end-date"]
    end
    if cmd_line_args["export-compartments-time-t"] !== nothing
        config["simulation"]["export_compartments_time_t"] = cmd_line_args["export-compartments-time-t"]
    end
    if cmd_line_args["export-compartments-full"] == true
        config["simulation"]["export_compartments_full"] = true
    end

    nothing
end


const OUTPUT_FORMATS = Dict("netcdf" => NetCDFFormat(), "hdf5" => HDF5Format())

get_output_format(output_format::String) = get(OUTPUT_FORMATS, output_format, NetCDFFormat())
get_output_format_str(output_format::AbstractOutputFormat) = findfirst(==(output_format), OUTPUT_FORMATS)

function save_full(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, output_format::Union{String,AbstractOutputFormat}; kwargs...)
    
    format = output_format isa String ? get_output_format(output_format) : output_format
    _save_full(engine, epi_params, population, output_path, format; kwargs...)
end

function _save_full(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, ::NetCDFFormat; 
    G_coords=String[], M_coords=String[], T_coords=String[])
    
    
    filename = joinpath(output_path, "compartments_full.nc")
    @info "- Saving full simulation output in NetCDF: $filename"
    try
        MMCACovid19Vac.save_simulation_netCDF(epi_params, population, filename; G_coords, M_coords, T_coords)
    catch e
        @error "Error saving simulation output" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "- Done saving"
end

function _save_full(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, ::HDF5Format; kwargs...)

    filename = joinpath(output_path, "compartments_full.h5")
    @info "- Saving full simulation output in HDF5: $filename"
    MMCACovid19Vac.save_simulation_hdf5(epi_params, population, filename)
end

function save_time_step(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, export_compartments_time_t::Int, export_date::Date) 
    
    filename = joinpath(output_path, "compartments_t_$(export_date).h5")

    @info "- Saving simulation state at time step: $(time_step_tosave) ($(export_date))"
    @info "\t- filename: $(filename)"
    MMCACovid19Vac.save_simulation_hdf5(epi_params, population, filename; 
                        export_time_t = export_compartments_time_t)
end

function save_observables(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String; 
    G_coords=String[], M_coords=String[], T_coords=String[])

    filename = joinpath(output_path, "observables.nc")
    @info "Storing simulation observables output in NetCDF: $filename"
    try
        MMCACovid19Vac.save_observables_netCDF(epi_params, population, filename; G_coords, M_coords, T_coords)
    catch e
        @error "Error saving simulation observables" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "done saving ??"
end




function save_full(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String, output_format::Union{String,AbstractOutputFormat}; kwargs...)
    format = output_format isa String ? get_output_format(output_format) : output_format
    _save_full(engine, epi_params, population, output_path, format; kwargs...)
end

function _save_full(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String, ::NetCDFFormat; G_coords=String[], M_coords=String[], T_coords=String[])
    
    
    filename = joinpath(output_path, "compartments_full.nc")
    @info "- Saving full simulation output in NetCDF: $filename"
    try
        G = population.G
        M = population.M
        T = epi_params.T

        if length(G_coords) != G
            G_coords = collect(1:G)
        end
        if length(M_coords) != M
            M_coords = collect(1:M)
        end
        if length(T_coords) != T
            T_coords = collect(1:T) 
        end
        
        g_dim = NcDim("G", G, atts=Dict("description" => "Age strata", "Unit" => "unitless"), values=G_coords, unlimited=false)
        m_dim = NcDim("M", M, atts=Dict("description" => "Region", "Unit" => "unitless"), values=M_coords, unlimited=false)
        t_dim = NcDim("T", T, atts=Dict("description" => "Time", "Unit" => "unitless"), values=T_coords, unlimited=false)
        dimlist = [g_dim, m_dim, t_dim]

        S  = NcVar("S" , dimlist; atts=atts=Dict("description" => "Suceptibles"), t=Float64, compress=-1)
        E  = NcVar("E" , dimlist; atts=atts=Dict("description" => "Exposed"), t=Float64, compress=-1)
        A  = NcVar("A" , dimlist; atts=atts=Dict("description" => "Asymptomatic"), t=Float64, compress=-1)
        I  = NcVar("I" , dimlist; atts=atts=Dict("description" => "Infected"), t=Float64, compress=-1)
        PH = NcVar("PH", dimlist; atts=atts=Dict("description" => "Pre-hospitalized"), t=Float64, compress=-1)
        PD = NcVar("PD", dimlist; atts=atts=Dict("description" => "Pre-deceased"), t=Float64, compress=-1)
        HR = NcVar("HR", dimlist; atts=atts=Dict("description" => "Hospitalized-good"), t=Float64, compress=-1)
        HD = NcVar("HD", dimlist; atts=atts=Dict("description" => "Hospitalized-bad"), t=Float64, compress=-1)
        R  = NcVar("R" , dimlist; atts=atts=Dict("description" => "Recovered"), t=Float64, compress=-1)
        D  = NcVar("D" , dimlist; atts=atts=Dict("description" => "Dead"), t=Float64, compress=-1)
        varlist = [S, E, A, I, PH, PD, HR, HD, R, D]

        data_dict = Dict()
        data_dict["S"]  = epi_params.ρˢᵍ  .* population.nᵢᵍ
        data_dict["E"]  = epi_params.ρᴱᵍ  .* population.nᵢᵍ
        data_dict["A"]  = epi_params.ρᴬᵍ  .* population.nᵢᵍ
        data_dict["I"]  = epi_params.ρᴵᵍ  .* population.nᵢᵍ
        data_dict["PH"] = epi_params.ρᴾᴴᵍ .* population.nᵢᵍ
        data_dict["PD"] = epi_params.ρᴾᴰᵍ .* population.nᵢᵍ
        data_dict["HR"] = epi_params.ρᴴᴿᵍ .* population.nᵢᵍ
        data_dict["HD"] = epi_params.ρᴴᴰᵍ .* population.nᵢᵍ
        data_dict["R"]  = epi_params.ρᴿᵍ  .* population.nᵢᵍ
        data_dict["D"]  = epi_params.ρᴰᵍ  .* population.nᵢᵍ

        # the next steps are needed to guarantee the the total population
        # remains constant and for this we need to calculate the number
        # of confined individuals at every time step and the sum that 
        # value to the suceptible comparment.

        sim_pop = ( epi_params.ρˢᵍ + epi_params.ρᴱᵍ + epi_params.ρᴬᵍ + 
                    epi_params.ρᴵᵍ + epi_params.ρᴾᴴᵍ + epi_params.ρᴾᴰᵍ + 
                    epi_params.ρᴴᴿᵍ + epi_params.ρᴴᴰᵍ + epi_params.ρᴿᵍ + 
                    epi_params.ρᴰᵍ ) .* population.nᵢᵍ
        
    
        @simd for t in 1:T
            CH = population.nᵢᵍ - sim_pop[:, :, t]
            data_dict["S"][:, :, t] .+= CH
        end

        isfile(filename) && rm(filename)

        NetCDF.create(filename, varlist, mode=NC_NETCDF4)
        for (var_label, data) in data_dict
            ncwrite(data, filename, var_label)
        end
    catch e
        @error "Error saving simulation output" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "- Done saving"

end

function create_compartments_array(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params)
    G = population.G
    M = population.M
    T = epi_params.T
    N = 10

    compartments = zeros(Float64, G, M, T, N);
    compartments[:, :, :, 1]  .= epi_params.ρˢᵍ .* population.nᵢᵍ
    compartments[:, :, :, 2]  .= epi_params.ρᴱᵍ .* population.nᵢᵍ
    compartments[:, :, :, 3]  .= epi_params.ρᴬᵍ .* population.nᵢᵍ
    compartments[:, :, :, 4]  .= epi_params.ρᴵᵍ .* population.nᵢᵍ
    compartments[:, :, :, 5]  .= epi_params.ρᴾᴴᵍ .* population.nᵢᵍ
    compartments[:, :, :, 6]  .= epi_params.ρᴾᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 7]  .= epi_params.ρᴴᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 8]  .= epi_params.ρᴴᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 9]  .= epi_params.ρᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 10] .= epi_params.ρᴰᵍ .* population.nᵢᵍ

    sim_pop = sum(compartments, dims=4)[:, :, :, 1]
    
    @simd for t in 1:T
        CH = population.nᵢᵍ - sim_pop[:, :, t]
        compartments[:, :, t, 1] += CH
    end

    return compartments
end

function _save_full(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String, ::HDF5Format; kwargs...)
    
    filename = joinpath(output_path, "compartments_full.h5")
    @info "- Storing full simulation output in HDF5: $filename"
    compartments = create_compartments_array(engine, epi_params, population)

    isfile(filename) && rm(filename)
    h5open(filename, "w") do file
        write(file, "data", compartments[:,:,:,:,:])
    end
end


function save_time_step(engine::MMCACovid19Engine,
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String, export_time_t::Int, export_date::Date) 
    
    filename = joinpath(output_path, "compartments_t_$(export_date).h5")
    @info "\t- filename: $(filename)"
    
    compartments = create_compartments_array(engine, epi_params, population)

    isfile(filename) && rm(filename)
    h5open(filename, "w") do file
        write(file, "data", compartments[:,:,export_time_t,:])
    end
end


function save_observables(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String; 
    
    G_coords=String[], M_coords=String[], T_coords=String[])

    filename = joinpath(output_path, "observables.nc")
    @info "- Storing simulation observables output in NetCDF: $filename"
    try
        G = population.G
        M = population.M
        T = epi_params.T
    
        if length(G_coords) != G
            G_coords = collect(1:G)
        end
        if length(M_coords) != M
            M_coords = collect(1:M)
        end
        if length(T_coords) != T
            T_coords = collect(1:T) 
        end
    
        g_dim = NcDim("G", G, atts=Dict("description" => "Age strata", "Unit" => "unitless"), values=G_coords, unlimited=false)
        m_dim = NcDim("M", M, atts=Dict("description" => "Region", "Unit" => "unitless"), values=M_coords, unlimited=false)
        t_dim = NcDim("T", T, atts=Dict("description" => "Time", "Unit" => "unitless"), values=T_coords, unlimited=false)
        dimlist = [g_dim, m_dim, t_dim]
    
        newI  = NcVar("new_infected" , dimlist; atts=atts=Dict("description" => "Suceptibles"), t=Float64, compress=-1)
        newH  = NcVar("new_hospitalized" , dimlist; atts=atts=Dict("description" => "Exposed"), t=Float64, compress=-1)
        newD  = NcVar("new_deaths" , dimlist; atts=atts=Dict("description" => "Asymptomatic"), t=Float64, compress=-1)
        varlist = [newI, newH, newD]
     
        data_dict = Dict()
        data_dict["new_infected"] = (epi_params.ρᴬᵍ  .* population.nᵢᵍ) .* epi_params.αᵍ
            
        hosp_rates = epi_params.μᵍ .* (1 .- epi_params.θᵍ) .* epi_params.γᵍ
        hosp_rates = reshape(hosp_rates, G, 1, 1)
            
        data_dict["new_hospitalized"] = ((epi_params.ρᴵᵍ  .* population.nᵢᵍ) .* hosp_rates)
            
        D = epi_params.ρᴰᵍ
        data_dict["new_deaths"] = zeros(size(D))
        data_dict["new_deaths"][:, :, 2:end] = diff((D .* population.nᵢᵍ), dims=3)
        
        isfile(filename) && rm(filename)
        NetCDF.create(filename, varlist, mode=NC_NETCDF4)
        for var_label in keys(data_dict)
            data = data_dict[var_label]
            ncwrite(data, filename, var_label)
        end
    catch e
        @error "Error saving simulation observables" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "- Done saving"
end