@defstruct ConfusionMatrixLayer Layer (
  name :: AbstractString = "confusion_matrix",
  report_error :: Bool = false,
  (dim :: Int = -2, dim != 0),
  (bottoms :: Vector{Symbol} = Symbol[], length(bottoms) == 2),
)
@characterize_layer(ConfusionMatrixLayer,
  is_sink    => true,
  has_stats  => true,
)

type ConfusionMatrixLayerState <: LayerState
  layer :: ConfusionMatrixLayer

  op_dim   :: Int
  dim      :: Int
  confusionMatrix :: SparseMatrixCSC{Int,Int}
  etc      :: Any
end

function setup_etc(backend::CPUBackend, layer::ConfusionMatrixLayer, op_dim::Int, inputs)
  nothing
end

function setup(backend::Backend, layer::ConfusionMatrixLayer, inputs::Vector{Blob}, diffs::Vector{Blob})
  total_dim = ndims(inputs[1])
  op_dim = layer.dim < 0 ? layer.dim + total_dim + 1 : layer.dim
  @assert 1 <= op_dim <= total_dim
  @assert op_dim != total_dim

  etc = setup_etc(backend, layer, op_dim, inputs)
  dim = size(inputs[1],1)
  return ConfusionMatrixLayerState(layer, op_dim, dim, spzeros(dim,dim), etc)
end
function shutdown(backend::CPUBackend, state::ConfusionMatrixLayerState)
end

function reset_statistics(state::ConfusionMatrixLayerState)
  state.confusionMatrix = spzeros(state.dim,state.dim)
end

function dump_statistics(storage, state::ConfusionMatrixLayerState, show::Bool)
  @info("Confusion Matrix Analaysis:" * state.layer.name)
  accuracy = sum(diag(state.confusionMatrix)) ./ sum(state.confusionMatrix)
  update_statistics(storage, "$(state.layer.name)-accuracy", accuracy)
  if state.layer.report_error
    update_statistics(storage, "$(state.layer.name)-error", 1-accuracy)
  end

  if show
    idx = find(sum(state.confusionMatrix,2))
    idy = find(sum(state.confusionMatrix,1))
    report = full(state.confusionMatrix[idx,idy])
    report = report ./ sum(report,1)
    report = hcat(idx-1,report)
    report = vcat(hcat(0,idy'-1),report)
    #report = full(state.confusionMatrix)
    dump(report)
  end
end

function forward(backend::CPUBackend, state::ConfusionMatrixLayerState, inputs::Vector{Blob})
  pred = inputs[1].data
  label = inputs[2].data
  dim_pre, dim_prob, dim_post = split_dims(pred, state.op_dim)
  accuracy = 0.0
  for i = 0:dim_pre-1
    for j = 0:dim_post-1
      idx = Int[i + dim_pre*(k + dim_prob*j) for k=0:(dim_prob-1)] + 1
      @inbounds state.confusionMatrix[round(Int,label[i + dim_pre*j + 1]) + 1, indmax(pred[idx])] += 1.0
      @inbounds if round(Int, label[i + dim_pre*j + 1])+1 == indmax(pred[idx])
        accuracy += 1.0
      end
    end
  end

  #state.accuracy = convert(Float64, state.accuracy * state.n_accum + accuracy) / (state.n_accum + length(label))
  #state.n_accum += length(label)
end

function backward(backend::Backend, state::ConfusionMatrixLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
end

