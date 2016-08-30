using PyPlot
#Note: This is a Pseudo ROC plot for Binary classification with 2 neural node output.
@defstruct ROCPlotLayer Layer (
  name :: AbstractString = "receiver_operator_characteristic-plot",
  report_error :: Bool = false,
  (dim :: Int = -2, dim != 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(ROCPlotLayer,
  is_sink    => true,
  has_stats  => true,
)

type ROCPlotLayerState <: LayerState
  layer :: ROCPlotLayer

  op_dim   :: Int
  x1        :: Array{Float32,1}
  x2        :: Array{Float32,1}
  etc      :: Any
end

function setup_etc(backend::CPUBackend, layer::ROCPlotLayer, op_dim::Int, inputs)
  nothing
end

function setup(backend::Backend, layer::ROCPlotLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  total_dim = ndims(inputs[1])
  op_dim = layer.dim < 0 ? layer.dim + total_dim + 1 : layer.dim
  @assert 1 <= op_dim <= total_dim
  @assert op_dim != total_dim
  
  etc = setup_etc(backend, layer, op_dim, inputs)
  return ROCPlotLayerState(layer, op_dim, Array{Float32,1}(), Array{Float32,1}(), etc)
end
function shutdown(backend::CPUBackend, state::ROCPlotLayerState)
end

function reset_statistics(state::ROCPlotLayerState)
  state.x1 = Array{Float32,1}()
  state.x2 = Array{Float32,1}()
end

function dump_statistics(storage, state::ROCPlotLayerState, show::Bool)
  @info("ROC Plot Analaysis:" * state.layer.name)
  accuracy = 0.0
  update_statistics(storage, "$(state.layer.name)-accuracy", accuracy)
  if state.layer.report_error
    update_statistics(storage, "$(state.layer.name)-error", 1-accuracy)
  end

  if show
    #TODO
    dump(length(state.x1))
    PyPlot.clf()
    maxval = reduce(max,state.x1)
    minval = reduce(min,state.x1)
    lx1 = length(state.x1)
    x = map(i -> sum(state.x1 .> i)/lx1 ,linspace(minval,maxval,1000))
    
    maxval = reduce(max,state.x2)
    minval = reduce(min,state.x2)
    lx2 = length(state.x2)
    y = map(i -> sum(state.x2 .> i)/lx2 ,linspace(minval,maxval,1000))

    plot(x,y, color="black",linewidth=2.0,linestyle="-")
    if isfile("roc-plot.svg")
        rm("roc-plot.svg")
    end
    savefig("roc-plot.svg")
  end
end

function forward(backend::CPUBackend, state::ROCPlotLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data
  

  dim_pre, dim_prob, dim_post = split_dims(pred, state.op_dim)
  accuracy = 0.0
  for i = 0:dim_pre-1
    for j = 0:dim_post-1
      idx = Int[i + dim_pre*(k + dim_prob*j) for k=0:(dim_prob-1)] + 1
      if round(Int, label[i + dim_pre*j + 1]) == 1
          push!(state.x1,pred[idx][1])
      else
          push!(state.x2,pred[idx][2])
      end
    end
  end

  #state.accuracy = convert(Float64, state.accuracy * state.n_accum + accuracy) / (state.n_accum + length(label))
  #state.n_accum += length(label)
end

function backward(backend::Backend, state::ROCPlotLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

