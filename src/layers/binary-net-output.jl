using PyPlot
@defstruct BinaryNetPlotLayer Layer (
  name :: AbstractString = "Binary-Net-plot",
  report_error :: Bool = false,
  (dim :: Int = -2, dim != 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(BinaryNetPlotLayer,
  is_sink    => true,
  has_stats  => true,
)

type BinaryNetPlotLayerState <: LayerState
  layer :: BinaryNetPlotLayer

  op_dim   :: Int
  x1        :: Array{Float32,1}
  y1        :: Array{Float32,1}
  x2        :: Array{Float32,1}
  y2        :: Array{Float32,1}
  etc      :: Any
end

function setup_etc(backend::CPUBackend, layer::BinaryNetPlotLayer, op_dim::Int, inputs)
  nothing
end

function setup(backend::Backend, layer::BinaryNetPlotLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  total_dim = ndims(inputs[1])
  op_dim = layer.dim < 0 ? layer.dim + total_dim + 1 : layer.dim
  @assert 1 <= op_dim <= total_dim
  @assert op_dim != total_dim
  
  etc = setup_etc(backend, layer, op_dim, inputs)
  return BinaryNetPlotLayerState(layer, op_dim, Array{Float32,1}(), Array{Float32,1}(), Array{Float32,1}(), Array{Float32,1}(), etc)
end
function shutdown(backend::CPUBackend, state::BinaryNetPlotLayerState)
end

function reset_statistics(state::BinaryNetPlotLayerState)
  state.x1 = Array{Float32,1}()
  state.y1 = Array{Float32,1}()
  state.x2 = Array{Float32,1}()
  state.y2 = Array{Float32,1}()
end

function dump_statistics(storage, state::BinaryNetPlotLayerState, show::Bool)
  @info("BinaryNet Plot Analaysis:" * state.layer.name)
  accuracy = 0.0
  update_statistics(storage, "$(state.layer.name)-accuracy", accuracy)
  if state.layer.report_error
    update_statistics(storage, "$(state.layer.name)-error", 1-accuracy)
  end

  if show
    dump(length(state.x1) + length(state.x2))
    PyPlot.clf()
    plot(state.x1,state.y1, color="blue",linewidth=0.0,linestyle="",marker=".", alpha=0.02)
    plot(state.x2,state.y2, color="red",linewidth=0.0,linestyle="",marker=".", alpha=0.02)
    plot(collect(0:10), collect(0:10), color="black",linewidth=1.0, linestyle="-")
    if isfile("bin-net-plot.svg")
        rm("bin-net-plot.svg")
    end
    savefig("bin-net-plot.svg")
  end
end

function forward(backend::CPUBackend, state::BinaryNetPlotLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data
  

  dim_pre, dim_prob, dim_post = split_dims(pred, state.op_dim)
  accuracy = 0.0
  for i = 0:dim_pre-1
    for j = 0:dim_post-1
      idx = Int[i + dim_pre*(k + dim_prob*j) for k=0:(dim_prob-1)] + 1
      if round(Int, label[i + dim_pre*j + 1]) == 1
          push!(state.x1,pred[idx][1])
          push!(state.y1,pred[idx][2])
      else
          push!(state.x2,pred[idx][1])
          push!(state.y2,pred[idx][2])
      end
    end
  end

  #state.accuracy = convert(Float64, state.accuracy * state.n_accum + accuracy) / (state.n_accum + length(label))
  #state.n_accum += length(label)
end

function backward(backend::Backend, state::BinaryNetPlotLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

