# Package installations (Run once per environment)
import Pkg
Pkg.add("MLUtils")
Pkg.add("ROCAnalysis")
Pkg.add("WordTokenizers")
Pkg.add("TextAnalysis")
Pkg.add("MLBase")
Pkg.add("Flux")
Pkg.add("MLJ")
Pkg.add("StatisticalMeasures")
Pkg.add("DataAPI")
Pkg.add("Plots")

# Load required libraries
using PlutoUI
using CSV, DataFrames
using Flux: LSTM
using Flux: DataLoader
 using Flux: Chain, Dense, reset!, LSTM, LSTMCell, Optimisers, Embedding, onehotbatch, onecold, crossentropy, softmax, Adam, logitcrossentropy, Optimisers
using MLUtils: DataLoader
using Statistics, Random
using ROCAnalysis: remove_missing, roc, auc
using WordTokenizers
using TextAnalysis, Unicode
using MLBase: precision, recall, f1score
using MLJ
using StatisticalMeasures
using DataAPI
using Plots
using Flux


# Load dataset
base_path = "C:/Users/Asser/Documents/NUST/AI/Assignment 3/Medical-Abstracts-TC-Corpus-main/Medical-Abstracts-TC-Corpus-main/"
train_df = CSV.read(base_path * "medical_tc_train.csv", DataFrame)
test_df = CSV.read(base_path * "medical_tc_test.csv", DataFrame)
labels_df = CSV.read(base_path * "medical_tc_labels.csv", DataFrame)
label_map = Dict(labels_df.condition_label .=> labels_df.condition_name)


# Clean and preprocess text
function clean_text(text::String)
    lowercase(strip(replace(text, r"[^\p{L}\s]" => "")))
end
train_df.text_clean = clean_text.(train_df.medical_abstract)
test_df.text_clean = clean_text.(test_df.medical_abstract)

# Tokenization and vocabulary mapping
tokenizer = WordTokenizers.tokenize
train_tokens = tokenizer.(train_df.text_clean)
all_tokens = collect(Iterators.flatten(train_tokens))
vocab = unique(all_tokens)
vocab_map = Dict(word => i for (i, word) in enumerate(vocab))
vocab_size = length(vocab)
pad_token = vocab_size + 1

# Encode token sequences to fixed length
function encode_sequence(tokens::Vector{String}, vocab_map::Dict{String, Int}, maxlen::Int)
    seq = [get(vocab_map, t, pad_token) for t in tokens]
    if length(seq) < maxlen
        return vcat(fill(pad_token, maxlen - length(seq)), seq)
    else
        return last(seq, maxlen)
    end
end

# Prepare input and output data
maxlen = 30
X_train = [encode_sequence(t, vocab_map, maxlen) for t in train_tokens]
X_train = hcat(X_train...) |> transpose
y_train = onehotbatch(train_df.condition_label, sort(unique(train_df.condition_label)))
num_classes = size(y_train, 1)

unique_labels_sorted = sort(unique(train_df.condition_label))

# Mini-batching using DataLoader
batch_size = 32
train_loader = DataLoader((X_train', y_train), batchsize=batch_size, shuffle=true)

# Define the model
embedding_dim = 64
num_classes = length(unique(train_df.condition_label))
model = Chain(
    Embedding(vocab_size + 1, embedding_dim),   
    LSTM(embedding_dim => 128),                 
    x -> x[:, end, :],                          
    Dense(128, num_classes),
    softmax
)


# Setup optimizer
opt_state = Flux.setup(Adam(0.001), model)

# Training loop
n_epochs = 3
@info "Training started"
for epoch in 1:n_epochs
    total_loss = 0.0
    for (x, y) in train_loader
        grads = Flux.gradient(model) do m
            ŷ = m(x)
            loss = logitcrossentropy(ŷ, y)
            total_loss += loss
            return loss
        end
        Flux.update!(opt_state, model, grads[1])
    end
    println("Epoch $epoch complete - Loss: $(round(total_loss, digits=4))")
end
@info "Training completed"

# Evaluation metrics
Flux.reset!(model)

y_pred = Int[]
y_true = Int[]

for (x, y) in train_loader
    ŷ = model(x)
    append!(y_pred, Int.(onecold(ŷ)))
    append!(y_true, Int.(onecold(y)))
end

y_pred_int = Int.(y_pred)
y_true_int = Int.(y_true)

labels = sort(unique(y_true))
acc = mean(y_pred_int .== y_true_int)

# Compute metrics
cm = confusion_matrix(y_true_int, y_pred_int)

prec = StatisticalMeasures.precision(cm)
rec = StatisticalMeasures.recall(cm)
f1 = StatisticalMeasures.f1score(cm) # And f1score

println("Accuracy: $(round(acc, digits=4))")
println("Precision: $(round(prec, digits=4))")
println("Recall: $(round(rec, digits=4))")
println("F1 Score: $(round(f1, digits=4))")

# ROC Curve for a selected class
unique_labels_sorted = sort(unique(train_df.condition_label))

target_class_idx = 1

target_class_value = unique_labels_sorted[target_class_idx]
target_class_name = label_map[target_class_value]

println("Generating ROC curve for class: $target_class_name (Index $target_class_idx)")

probabilities_for_target_class = Float64[]
true_binary_labels = Bool[]

for (x_batch, y_batch) in train_loader
    ŷ_batch = model(x_batch) 
    batch_probs = ŷ_batch[target_class_idx, :]
    batch_labels = y_batch[target_class_idx, :] .> 0.5
    
    # Process each element in the batch
    for i in 1:length(batch_probs)
        # Skip missing values
        if !ismissing(batch_probs[i]) && !ismissing(batch_labels[i])
            push!(probabilities_for_target_class, batch_probs[i])
            push!(true_binary_labels, batch_labels[i])
        end
    end
end

# Compute and plot ROC curve
true_binary_numeric = Float64.(true_binary_labels)  # Convert Bool to Float64
r = roc(probabilities_for_target_class, true_binary_numeric)
auc_val = auc(r)  # Compute AUC

plot(r,
     title = "ROC Curve for Class $(target_class_value) | AUC = $(round(auc_val, digits=4))",
     xlabel = "False Positive Rate",
     ylabel = "True Positive Rate",
     lw = 2,
     label = false)