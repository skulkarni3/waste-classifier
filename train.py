import torch
from torch.utils.data import DataLoader
from waste_classifier import WasteClassifierCNN, CombinedWasteDataset, transform, compute_class_weights
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from datetime import datetime
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(waste_classification_path, household_waste_path, garbage_dataset_path, realwaste_path, use_only_landfill_from_new=False):
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_dir = os.path.join(models_dir, f'run_{timestamp}')
    os.makedirs(model_save_dir, exist_ok=True)

    # Start MLflow run
    with mlflow.start_run(run_name=f"waste_classifier_{timestamp}") as run:
        # Log model parameters
        mlflow.log_param("model_type", "CNN")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("weight_decay", 1e-4)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("num_epochs", 20)
        mlflow.log_param("early_stopping_patience", 5)
        mlflow.log_param("use_only_landfill_from_new", use_only_landfill_from_new)
        
        # Create datasets and log dataset sizes
        train_dataset = CombinedWasteDataset(
            waste_classification_path=waste_classification_path,
            household_waste_path=household_waste_path,
            garbage_dataset_path=garbage_dataset_path,
            realwaste_path=realwaste_path,
            transform=transform,
            train=True,
            use_only_landfill_from_new=use_only_landfill_from_new
        )
        
        val_dataset = CombinedWasteDataset(
            waste_classification_path=waste_classification_path,
            household_waste_path=household_waste_path,
            garbage_dataset_path=garbage_dataset_path,
            realwaste_path=realwaste_path,
            transform=transform,
            train=False,
            use_only_landfill_from_new=use_only_landfill_from_new
        )
        
        # Compute class weights
        class_weights = compute_class_weights(train_dataset)
        print("\nClass weights:", class_weights)
        mlflow.log_param("class_weights", class_weights.tolist())
        
        mlflow.log_param("train_size", len(train_dataset))
        mlflow.log_param("val_size", len(val_dataset))
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=8
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8
        )
        
        # Initialize model, loss function, and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WasteClassifierCNN().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
        
        # Training loop
        num_epochs = 20
        best_val_acc = 0
        patience = 5
        no_improve = 0
        
        print(f"Model checkpoints will be saved in: {model_save_dir}")
        print(f"Training on device: {device}")
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Print progress every 100 batches
                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            class_correct = [0] * 3
            class_total = [0] * 3
            
            # Lists to store predictions and true labels for F1 score
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Store predictions and labels for F1 score
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Per-class accuracy
                    for i in range(len(labels)):
                        label = labels[i]
                        pred = predicted[i]
                        class_correct[label] += (pred == label).item()
                        class_total[label] += 1
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            # Calculate F1 score
            f1 = f1_score(all_labels, all_preds, average='weighted')
            
            # Calculate confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "f1_score": f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Log per-class accuracies
            classes = ['Recyclable', 'Compostable', 'Landfill']
            for i in range(3):
                acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                mlflow.log_metric(f"{classes[i]}_accuracy", acc, step=epoch)
            
            # Print metrics
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'F1 Score: {f1:.4f}')
            print('\nPer-class accuracy:')
            for i in range(3):
                acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                print(f'{classes[i]}: {acc:.2f}%')
            
            # Print confusion matrix
            print('\nConfusion Matrix:')
            print('Predicted →')
            print('Actual ↓')
            print('          Recyclable  Compostable  Landfill')
            for i, class_name in enumerate(classes):
                row = f'{class_name:10}'
                for j in range(3):
                    row += f'{cm[i][j]:12}'
                print(row)
            print('-' * 50)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save checkpoint every 5 epochs or if it's the best model
            if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
                checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'f1_score': f1,
                    'timestamp': timestamp
                }, checkpoint_path)
                
                # Save confusion matrix plot
                cm_plot_path = os.path.join(model_save_dir, f'confusion_matrix_epoch_{epoch+1}.png')
                plot_confusion_matrix(cm, classes, cm_plot_path)
                
                # Log confusion matrix plot to MLflow
                mlflow.log_artifact(cm_plot_path)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save best model info
                    info_path = os.path.join(model_save_dir, 'training_info.txt')
                    with open(info_path, 'w') as f:
                        f.write(f"Best validation accuracy: {val_acc:.2f}%\n")
                        f.write(f"Achieved at epoch: {epoch + 1}\n")
                        f.write(f"Training accuracy: {train_acc:.2f}%\n")
                        f.write(f"Validation loss: {val_loss:.4f}\n")
                        f.write(f"Training loss: {train_loss:.4f}\n")
                        f.write(f"F1 Score: {f1:.4f}\n")
                        f.write(f"Model saved at: {checkpoint_path}\n")
                        f.write(f"MLflow run ID: {run.info.run_id}\n")
                    
                    # Log model to MLflow
                    mlflow.pytorch.log_model(model, "model")
                    
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    print(f'Best validation accuracy: {best_val_acc:.2f}%')
                    print(f'Model saved at: {os.path.join(model_save_dir, "best_model.pth")}')
                    break
        
        # Log final best metrics
        mlflow.log_metrics({
            "best_val_accuracy": best_val_acc,
            "final_train_accuracy": train_acc,
            "final_val_loss": val_loss,
            "final_train_loss": train_loss,
            "total_epochs": epoch + 1
        })
        
    return model_save_dir

if __name__ == "__main__":
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:////Users/shruti/USF-Spring-2/deep_learning/mlflow.db")
    mlflow.set_experiment("waste_classifier")
    
    # Paths to your datasets
    waste_classification_path = "data/waste-classification"
    household_waste_path = "data/household-waste/images/images"
    garbage_dataset_path = "data/garbage-dataset"
    realwaste_path = "data/realwaste"
    
    # Train from scratch
    save_dir = train_model(
        waste_classification_path,
        household_waste_path,
        garbage_dataset_path,
        realwaste_path,
        use_only_landfill_from_new=True
    )
    print(f"\nTraining completed! Model and training info saved in: {save_dir}")
    print("View MLflow UI with: mlflow ui") 