import Pkg;
Pkg.add("ThreadsX")
using Flux, Images, ImageTransformations, FileIO, Statistics, Plots
using Flux: onehotbatch, onecold, crossentropy
using MLBase: roc, auc, confusion_matrix, precision, recall, f1score
using ThreadsX, Serialization, ProgressMeter
using CUDA  # For GPU acceleration

# Check for GPU availability and set device
if CUDA.functional()
    @info "CUDA GPU detected - using GPU acceleration"
    device = gpu
else
    @warn "No CUDA GPU detected - falling back to CPU"
    device = cpu
end

# Configuration
const CLASSES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
const DATASET_PATH = "C:\\Users\\Asser\\Documents\\NUST\\AI\\Assignment 3\\Rice_Image_Dataset"
const CACHE_FILE = joinpath(DATASET_PATH, "preprocessed_cache.jls")
const IMG_SIZE = (224, 224)
const BATCH_SIZE = 32
const EPOCHS = 10
const LEARNING_RATE = 0.0001

# Data Preprocessing with Caching and Parallel Loading ------------------------
function preprocess_image(img_path)
    try
        img = load(img_path)
        img_gray = Gray.(img)
        img_resized = imresize(img_gray, IMG_SIZE)
        img_float = Float32.(channelview(img_resized))
        img_norm = (img_float .- minimum(img_float)) ./ (maximum(img_float) - minimum(img_float))
        return reshape(img_norm, size(img_norm, 1), size(img_norm, 2), 1)
    catch e
        @warn "Failed to process $img_path: $e"
        return nothing
    end
end

function load_dataset(;use_cache=true)
    if use_cache && isfile(CACHE_FILE)
        @info "Loading cached dataset..."
        return deserialize(CACHE_FILE)
    end

    @info "Preprocessing dataset (this may take a while)..."
    
    # Collect all image paths and labels
    image_paths = String[]
    labels = Int[]
    for (class_idx, class_name) in enumerate(CLASSES)
        class_path = joinpath(DATASET_PATH, class_name)
        files = readdir(class_path)
        append!(image_paths, joinpath.(class_path, files))
        append!(labels, fill(class_idx, length(files)))
    end

    # Parallel preprocessing with progress bar
    X = []
    Y = []
    progress = Progress(length(image_paths), 1, "Processing images...")
    
    results = ThreadsX.map(image_paths) do img_path
        img = preprocess_image(img_path)
        next!(progress)
        img
    end

    # Filter valid images
    valid_idx = findall(x -> x !== nothing, results)
    X = cat(results[valid_idx]..., dims=4)
    Y = labels[valid_idx]

    # Train-test split
    indices = shuffle(1:length(Y))
    split_idx = floor(Int, 0.8 * length(Y))
    
    train_data = (X[:, :, :, indices[1:split_idx]], Y[indices[1:split_idx]])
    test_data = (X[:, :, :, indices[split_idx+1:end]], Y[indices[split_idx+1:end]])

    # Cache the dataset
    if use_cache
        @info "Caching preprocessed dataset..."
        serialize(CACHE_FILE, (train_data, test_data))
    end

    return train_data, test_data
end

# Data Loaders ---------------------------------------------------------------
function create_dataloader(X, Y; batchsize=BATCH_SIZE, shuffle=false)
    # Move data to appropriate device (GPU/CPU)
    X_device = device(X)
    Y_device = device(Y)
    return DataLoader((X_device, Y_device), 
                    batchsize=batchsize, 
                    shuffle=shuffle,
                    partial=false,
                    parallel=true)
end

# Model Definition -----------------------------------------------------------
function create_model()
    model = Chain(
        # Input: 224x224x1
        Conv((11, 11), 1=>96, stride=4, relu),
        MaxPool((3, 3), stride=2),
        
        Conv((5, 5), 96=>256, pad=2, relu),
        MaxPool((3, 3), stride=2),
        
        Conv((3, 3), 256=>384, pad=1, relu),
        Conv((3, 3), 384=>384, pad=1, relu),
        Conv((3, 3), 384=>256, pad=1, relu),
        MaxPool((3, 3), stride=2),
        
        Flux.flatten,
        Dense(256*6*6, 4096, relu),
        Dropout(0.5),
        Dense(4096, 4096, relu),
        Dropout(0.5),
        Dense(4096, length(CLASSES))  # No activation for crossentropy
    ) |> device
    return model
end

# Training -------------------------------------------------------------------
function train!(model, train_loader, test_loader; epochs=EPOCHS)
    # Loss function and optimizer
    loss(x, y) = crossentropy(model(x), onehotbatch(y, 1:length(CLASSES)))
    opt = ADAM(LEARNING_RATE)
    
    # Training loop
    for epoch in 1:epochs
        # Training phase
        train_loss = 0.0
        progress = Progress(length(train_loader), 1, "Epoch $epoch Training...")
        
        for (x, y) in train_loader
            grads = gradient(Flux.params(model)) do
                l = loss(x, y)
                train_loss += l
                return l
            end
            Flux.update!(opt, Flux.params(model), grads)
            next!(progress)
        end
        
        # Evaluation phase
        test_acc, test_loss = evaluate(model, test_loader)
        avg_train_loss = train_loss / length(train_loader)
        
        # Corrected @info statement - all on one line
        @info "Epoch $epoch" avg_train_loss=round(avg_train_loss, digits=4) test_loss=round(test_loss, digits=4) test_acc=round(test_acc*100, digits=2)
    end
end
# Evaluation ----------------------------------------------------------------
function evaluate(model, loader)
    total_loss = 0.0
    correct = 0
    total = 0
    
    for (x, y) in loader
        logits = model(x)
        total_loss += crossentropy(logits, onehotbatch(y, 1:length(CLASSES)))
        correct += sum(onecold(logits) .== y)
        total += length(y)
    end
    
    accuracy = correct / total
    avg_loss = total_loss / length(loader)
    return accuracy, avg_loss
end

function compute_metrics(model, X, Y)
    # Move data to appropriate device
    X_device = device(X)
    Y_device = Y  # Labels stay on CPU
    
    # Get predictions
    logits = model(X_device) |> cpu
    probs = softmax(logits)
    preds = onecold(logits)
    
    # Confusion matrix
    cm = confusion_matrix(Y_device, preds)
    
    # Calculate metrics
    acc = sum(Y_device .== preds) / length(Y_device)
    precisions = [precision(cm, i) for i in 1:length(CLASSES)]
    recalls = [recall(cm, i) for i in 1:length(CLASSES)]
    f1s = [f1score(cm, i) for i in 1:length(CLASSES)]
    
    # ROC and AUC
    aucs = Float64[]
    for class_idx in 1:length(CLASSES)
        binary_true = Y_device .== class_idx
        class_probs = probs[class_idx, :]
        roc_curve = roc(binary_true, class_probs)
        push!(aucs, auc(roc_curve))
    end
    
    return (accuracy=acc, 
            precision=mean(precisions), 
            recall=mean(recalls), 
            f1=mean(f1s), 
            auc=mean(aucs),
            confusion_matrix=cm)
end

# Visualization -------------------------------------------------------------
function plot_confusion_matrix(cm, title)
    hm = heatmap(CLASSES, CLASSES, cm.matrix,
                title=title,
                xlabel="Predicted",
                ylabel="Actual",
                color=:blues)
    annotate!([(j, i, text(string(cm.matrix[i,j]), 8))  # Added closing parenthesis here
              for i in 1:length(CLASSES) for j in 1:length(CLASSES)])
    return hm
end

function plot_roc_curves(model, X, Y)
    X_device = device(X)
    logits = model(X_device) |> cpu
    probs = softmax(logits)
    
    plt = plot(layout=(length(CLASSES), 1), size=(800, 1200))
    aucs = Float64[]
    
    for class_idx in 1:length(CLASSES)
        binary_true = Y .== class_idx
        class_probs = probs[class_idx, :]
        roc_curve = roc(binary_true, class_probs)
        push!(aucs, auc(roc_curve))
        
        plot!(plt[class_idx], roc_curve.fp, roc_curve.tp,
              label="$(CLASSES[class_idx]) (AUC=$(round(aucs[end], digits=3)))",
              title="ROC: $(CLASSES[class_idx])",
              linewidth=2)
    end
    
    return plt, aucs
end

# Main Execution ------------------------------------------------------------
function main()
    # Load and preprocess data
    (train_X, train_Y), (test_X, test_Y) = load_dataset()
    
    # Create data loaders
    train_loader = create_dataloader(train_X, train_Y, shuffle=true)
    test_loader = create_dataloader(test_X, test_Y)
    
    # Initialize model
    model = create_model()
    @info "Model created" device=typeof(model[1].weight)  # Show where model is
    
    # Train the model
    @time train!(model, train_loader, test_loader)
    
    # Full evaluation
    metrics = compute_metrics(model, test_X, test_Y)
    
    # Print metrics
    println("\nFinal Evaluation Metrics:")
    println("Accuracy:    $(round(metrics.accuracy*100, digits=2))%")
    println("Precision:   $(round(metrics.precision*100, digits=2))%")
    println("Recall:      $(round(metrics.recall*100, digits=2))%")
    println("F1-Score:    $(round(metrics.f1*100, digits=2))%")
    println("AUC:         $(round(metrics.auc, digits=4))")
    
    # Visualizations
    cm_plot = plot_confusion_matrix(metrics.confusion_matrix, 
                                  "Confusion Matrix (Accuracy: $(round(metrics.accuracy*100, digits=2))%)")
    savefig(cm_plot, "confusion_matrix.png")
    
    roc_plot, _ = plot_roc_curves(model, test_X, test_Y)
    savefig(roc_plot, "roc_curves.png")
    
    return model, metrics
end

# Run the main function
model, metrics = main()