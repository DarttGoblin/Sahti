import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import warnings
warnings.filterwarnings('ignore')

def extract_subject_id(filename):
    """
    Extract subject ID from filename
    Abnormal: X_subjectnumber_X_X.wav
    Normal: N_subjectnumber_X_X.wav
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 2:
        return parts[1]  # subjectnumber is the second part
    else:
        # Fallback: use the entire filename without extension
        return os.path.splitext(basename)[0]
    
def load_dataset_from_directory(data_dir):
    """
    Load dataset from directory structure with subject information:
    data_dir/
        normal/
            N_subject001_X_X.wav
            N_subject002_X_X.wav
        abnormal/
            X_subject003_X_X.wav
            X_subject004_X_X.wav
    """
    audio_paths = []
    labels = []
    subjects = []
    
    # Map folder names to class indices
    # normal = 0, abnormal = 1
    class_mapping = {'normal': 0, 'abnormal': 1}
    
    for class_name, class_idx in class_mapping.items():
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            print(f"Loading {class_name} samples from {class_dir}")
            count = 0
            for filename in os.listdir(class_dir):
                if filename.endswith('.wav'):
                    subject_id = extract_subject_id(filename)
                    audio_paths.append(os.path.join(class_dir, filename))
                    labels.append(class_idx)
                    subjects.append(subject_id)
                    count += 1
            print(f"Found {count} {class_name} samples")
        else:
            print(f"Warning: Directory {class_dir} not found!")
    
    print(f"Total samples loaded: {len(audio_paths)}")
    print(f"Normal samples: {labels.count(0)}")
    print(f"Abnormal samples: {labels.count(1)}")
    
    # Print subject distribution
    unique_subjects = list(set(subjects))
    print(f"Total unique subjects: {len(unique_subjects)}")
    
    # Show subject distribution by class
    normal_subjects = set()
    abnormal_subjects = set()
    for i, (subject, label) in enumerate(zip(subjects, labels)):
        if label == 0:  # normal
            normal_subjects.add(subject)
        else:  # abnormal
            abnormal_subjects.add(subject)
    
    print(f"Subjects with normal recordings: {len(normal_subjects)}")
    print(f"Subjects with abnormal recordings: {len(abnormal_subjects)}")
    
    # Check for subjects appearing in both classes
    overlap = normal_subjects.intersection(abnormal_subjects)
    if overlap:
        print(f"Warning: {len(overlap)} subjects appear in both normal and abnormal classes: {overlap}")
    
    return audio_paths, labels, subjects

def subject_based_split(audio_paths, labels, subjects, test_size=0.2, random_state=42):
    """
    Split data by subjects to prevent data leakage
    """
    import numpy as np
    from collections import defaultdict
    
    # Group samples by subject and class
    subject_data = defaultdict(lambda: {'normal': [], 'abnormal': []})
    
    for i, (path, label, subject) in enumerate(zip(audio_paths, labels, subjects)):
        class_name = 'normal' if label == 0 else 'abnormal'
        subject_data[subject][class_name].append(i)
    
    # Separate subjects into different categories
    normal_only_subjects = []
    abnormal_only_subjects = []
    mixed_subjects = []
    
    for subject in subject_data:
        has_normal = len(subject_data[subject]['normal']) > 0
        has_abnormal = len(subject_data[subject]['abnormal']) > 0
        
        if has_normal and has_abnormal:
            mixed_subjects.append(subject)
        elif has_normal:
            normal_only_subjects.append(subject)
        elif has_abnormal:
            abnormal_only_subjects.append(subject)
    
    print(f"\nSubject distribution:")
    print(f"Normal-only subjects: {len(normal_only_subjects)}")
    print(f"Abnormal-only subjects: {len(abnormal_only_subjects)}")
    print(f"Unique Normal-only subjects: {len(set(normal_only_subjects))}")
    print(f"Unique Abnormal-only subjects: {len(set(abnormal_only_subjects))}")
    print(f"Mixed subjects (both normal and abnormal): {len(mixed_subjects)}")
    
    # Create stratified split by subject categories
    np.random.seed(random_state)
    
    # Split each category
    def split_subjects(subject_list, test_ratio):
        n_test = max(1, int(len(subject_list) * test_ratio))
        shuffled = subject_list.copy()
        np.random.shuffle(shuffled)
        return shuffled[:-n_test], shuffled[-n_test:]
    
    # Split subjects
    train_subjects = []
    val_subjects = []
    
    if normal_only_subjects:
        train_normal, val_normal = split_subjects(normal_only_subjects, test_size)
        train_subjects.extend(train_normal)
        val_subjects.extend(val_normal)
    
    if abnormal_only_subjects:
        train_abnormal, val_abnormal = split_subjects(abnormal_only_subjects, test_size)
        train_subjects.extend(train_abnormal)
        val_subjects.extend(val_abnormal)
    
    if mixed_subjects:
        train_mixed, val_mixed = split_subjects(mixed_subjects, test_size)
        train_subjects.extend(train_mixed)
        val_subjects.extend(val_mixed)
    
    # Convert subject splits to sample indices
    train_indices = []
    val_indices = []
    
    for i, subject in enumerate(subjects):
        if subject in train_subjects:
            train_indices.append(i)
        elif subject in val_subjects:
            val_indices.append(i)
    
    # Create splits
    train_paths = [audio_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_subjects_list = [subjects[i] for i in train_indices]
    
    val_paths = [audio_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_subjects_list = [subjects[i] for i in val_indices]
    
    # Verify no subject overlap
    train_subject_set = set(train_subjects_list)
    val_subject_set = set(val_subjects_list)
    overlap = train_subject_set.intersection(val_subject_set)
    
    if overlap:
        print(f"ERROR: Subject overlap detected: {overlap}")
        raise ValueError("Subject overlap in train/validation split!")
    else:
        print(f"✓ No subject overlap between train and validation sets")
    
    print(f"\nTrain subjects: {len(train_subject_set)}")
    print(f"Validation subjects: {len(val_subject_set)}")
    
    return train_paths, val_paths, train_labels, val_labels

class HeartbeatAudioDataset(Dataset):
    def __init__(self, audio_paths, labels, sample_rate=16000, duration=5.0, feature_extractor=None):
        """
        Heartbeat audio dataset for two-class classification
        
        Args:
            audio_paths: List of paths to audio files
            labels: List of labels (0=normal, 1=abnormal)
            sample_rate: Target sample rate for pre-trained models
            duration: Fixed duration in seconds
            feature_extractor: Hugging Face feature extractor
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.feature_extractor = feature_extractor
        self.target_length = int(sample_rate * duration)
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        try:
            # Load audio
            waveform, sr = torchaudio.load(self.audio_paths[idx])
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Flatten to 1D
            waveform = waveform.squeeze(0)
            
            # Pad or truncate to fixed length
            if waveform.shape[0] > self.target_length:
                # For heartbeat sounds, take center portion to preserve the main heartbeat
                start_idx = (waveform.shape[0] - self.target_length) // 2
                waveform = waveform[start_idx:start_idx + self.target_length]
            else:
                padding = self.target_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (padding//2, padding - padding//2))
            
            # Normalize audio
            if torch.std(waveform) > 0:
                waveform = (waveform - torch.mean(waveform)) / torch.std(waveform)
            
            #return waveform, torch.tensor(self.labels[idx], dtype=torch.long), other_features (torch.tensor)
            return waveform, torch.tensor(self.labels[idx], dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading {self.audio_paths[idx]}: {e}")
            # Return a zero tensor as fallback
            return torch.zeros(self.target_length), torch.tensor(self.labels[idx], dtype=torch.long)

class PretrainedAudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", freeze_encoder=True):
        """
        Pre-trained audio encoder for feature extraction
        
        Args:
            model_name: Pre-trained model to use
            freeze_encoder: Whether to freeze the encoder weights
        """
        super().__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        
        # Load pre-trained model
        print(f"Loading pre-trained model: {model_name}")
        self.encoder = Wav2Vec2Model.from_pretrained(model_name, cache_dir="/users/local/y17bendo/cache")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, cache_dir="/users/local/y17bendo/cache")
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder weights frozen for linear probing")
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz
            dummy_output = self.encoder(dummy_input).last_hidden_state
            self.feature_dim = dummy_output.shape[-1]
            print(f"Feature dimension: {self.feature_dim}")
    
    def forward(self, x):
        """
        Extract features from audio
        
        Args:
            x: Raw audio waveform [batch_size, sequence_length]
        
        Returns:
            features: Extracted features [batch_size, feature_dim]
        """
        with torch.set_grad_enabled(not self.freeze_encoder):
            # Extract features using pre-trained encoder
            outputs = self.encoder(x)
            features = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
            
            # Global average pooling over time dimension
            features = torch.mean(features, dim=1)  # [batch_size, hidden_dim]
            
        return features

class HeartbeatLinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout_rate=0.3):
        """
        Linear probe classifier for heartbeat classification
        
        Args:
            input_dim: Input feature dimension from pre-trained encoder
            num_classes: Number of classes (2 for normal/abnormal)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 256), ###### 256 + number of other features
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64), #### 256 + number of other features
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

def create_data_loaders(train_paths, train_labels, val_paths, val_labels, 
                       batch_size=16, sample_rate=16000, duration=5.0):
    """Create train and validation data loaders for heartbeat audio"""
    
    train_dataset = HeartbeatAudioDataset(
        train_paths, train_labels, 
        sample_rate=sample_rate, 
        duration=duration
    )
    
    val_dataset = HeartbeatAudioDataset(
        val_paths, val_labels,
        sample_rate=sample_rate,
        duration=duration
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced for stability
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader

def extract_features_with_pretrained(data_loader, encoder, device):
    """Extract features using pre-trained encoder"""
    features = []
    labels = []
    
    encoder.eval()
    print("Extracting features using pre-trained encoder...")
    
    with torch.no_grad():
        # for batch_audio, batch_labels, batch_other_features in tqdm(data_loader, desc="Extracting features"):
        for batch_audio, batch_labels in tqdm(data_loader, desc="Extracting features"):
            batch_audio = batch_audio.to(device)
            
            try:
                # Extract features using pre-trained encoder
                batch_features = encoder(batch_audio)
                batch_size = batch_features.shape[0]

                ###### Add the additional feature to the batch_features instead of randint
                # batch_features = torch.cat([batch_features.cpu(), torch.randint(0, 1, (batch_size, 4))], dim=1)
                features.append(batch_features.cpu())
                labels.append(batch_labels)
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Skip this batch
                continue
    
    if not features:
        raise ValueError("No features extracted! Check your audio files and model.")
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    print(f"Extracted features shape: {features.shape}")
    return features, labels

def train_heartbeat_classifier(features, labels, val_features, val_labels, 
                              num_epochs=200, learning_rate=0.001, device='cpu'):
    """Train the heartbeat classifier"""
    
    input_dim = features.shape[1]
    model = HeartbeatLinearProbe(input_dim, num_classes=2).to(device)
    
    # Use class weights to handle potential imbalance
    class_counts = torch.bincount(labels)
    class_weights = len(labels) / (2 * class_counts.float())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=15, factor=0.5, verbose=True
    )
    
    # Create data loaders for features
    feature_dataset = torch.utils.data.TensorDataset(features, labels)
    val_feature_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(feature_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_feature_dataset, batch_size=32, shuffle=False)
    
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 25
    
    train_losses = []
    val_accuracies = []
    
    print("Training heartbeat classifier...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                predictions = torch.argmax(outputs, dim=1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_true.extend(batch_labels.numpy())
        
        val_acc = accuracy_score(val_true, val_predictions)
        scheduler.step(val_acc)
        
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_heartbeat_classifier.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0 or epoch < 10:
            print(f'Epoch {epoch}/{num_epochs}: '
                  f'Train Loss: {epoch_loss/len(train_loader):.4f}, '
                  f'Val Acc: {val_acc:.4f}, '
                  f'Best Val Acc: {best_val_acc:.4f}')
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch} (patience: {max_patience})")
            break
    
    return model, train_losses, val_accuracies, best_val_acc

def evaluate_model(model, val_features, val_labels, device):
    """Evaluate the trained model"""
    model.eval()
    
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, report, cm

# def load_dataset_from_directory(data_dir):
#     """
#     Load dataset from directory structure:
#     data_dir/
#         normal/
#             audio1.wav
#             audio2.wav
#         abnormal/
#             audio3.wav
#             audio4.wav
#     """
#     audio_paths = []
#     labels = []
    
#     # Map folder names to class indices
#     # normal = 0, abnormal = 1
#     class_mapping = {'normal': 0, 'abnormal': 1}
    
#     for class_name, class_idx in class_mapping.items():
#         class_dir = os.path.join(data_dir, class_name)
#         if os.path.exists(class_dir):
#             print(f"Loading {class_name} samples from {class_dir}")
#             count = 0
#             for filename in os.listdir(class_dir):
#                 if filename.endswith('.wav'):
#                     audio_paths.append(os.path.join(class_dir, filename))
#                     labels.append(class_idx)
#                     count += 1
#             print(f"Found {count} {class_name} samples")
#         else:
#             print(f"Warning: Directory {class_dir} not found!")
    
#     print(f"Total samples loaded: {len(audio_paths)}")
#     print(f"Normal samples: {labels.count(0)}")
#     print(f"Abnormal samples: {labels.count(1)}")
    
#     return audio_paths, labels

def main():
    parser = argparse.ArgumentParser(description='Heartbeat audio classification with pre-trained embeddings')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Directory containing normal/ and abnormal/ folders')
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base',
                       choices=['facebook/wav2vec2-base', 'facebook/wav2vec2-large', 
                               'facebook/wav2vec2-large-960h', 'microsoft/wavlm-base-plus'],
                       help='Pre-trained model to use for feature extraction')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (reduced for memory efficiency)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Audio sample rate (must match pre-trained model)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Fixed audio duration in seconds')
    parser.add_argument('--freeze_encoder', action='store_true', default=True,
                       help='Freeze pre-trained encoder weights (linear probing)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Pre-trained model: {args.model_name}')
    
    # Load heartbeat data
    print('Loading heartbeat dataset...')
    audio_paths, labels, subjects = load_dataset_from_directory(args.data_dir)
    
    if len(audio_paths) == 0:
        print("No audio files found! Please check your data directory structure.")
        return
    
    # Split data (80% train, 20% validation) with stratification
    from sklearn.model_selection import train_test_split
    
    # train_paths, val_paths, train_labels, val_labels = train_test_split(
    #     audio_paths, labels, 
    #     test_size=0.2, 
    #     stratify=labels,  # Ensure balanced split
    #     random_state=42
    # )

    # Subject-based split to prevent data leakage
    print('\nPerforming subject-based train-validation split...')
    train_paths, val_paths, train_labels, val_labels = subject_based_split(
        audio_paths, labels, subjects, test_size=0.2, random_state=42
    )
    
    print(f'Training samples: {len(train_paths)} (Normal: {train_labels.count(0)}, Abnormal: {train_labels.count(1)})')
    print(f'Validation samples: {len(val_paths)} (Normal: {val_labels.count(0)}, Abnormal: {val_labels.count(1)})')
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels,
        batch_size=args.batch_size, sample_rate=args.sample_rate, 
        duration=args.duration
    )
    
    # Initialize pre-trained audio encoder
    try:
        encoder = PretrainedAudioEncoder(
            model_name=args.model_name, 
            freeze_encoder=args.freeze_encoder
        ).to(device)
    except Exception as e:
        print(f"Error loading pre-trained model: {e}")
        print("Please ensure you have transformers library installed: pip install transformers")
        return
    
    # Extract features using pre-trained encoder
    print('Extracting training features...')
    train_features, train_labels_tensor = extract_features_with_pretrained(train_loader, encoder, device)
    
    print('Extracting validation features...')
    val_features, val_labels_tensor = extract_features_with_pretrained(val_loader, encoder, device)
    
    print(f'Feature dimension: {train_features.shape[1]}')
    
    # Train heartbeat classifier
    print('Training heartbeat classifier...')
    model, train_losses, val_accuracies, best_val_acc = train_heartbeat_classifier(
        train_features, train_labels_tensor,
        val_features, val_labels_tensor,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
    )
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_heartbeat_classifier.pth'))
    accuracy, report, cm = evaluate_model(model, val_features, val_labels_tensor, device)
    
    print(f'\n=== HEARTBEAT CLASSIFICATION RESULTS ===')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    print(f'Final Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(report)
    print('\nConfusion Matrix:')
    print('[[TN, FP],')
    print(' [FN, TP]]')
    print(cm)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for abnormal class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for normal class
    
    print(f'\nAdditional Metrics:')
    print(f'Sensitivity (Abnormal Detection Rate): {sensitivity:.4f}')
    print(f'Specificity (Normal Detection Rate): {specificity:.4f}')
    
    # Save results
    results = {
        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'model_name': args.model_name,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'freeze_encoder': args.freeze_encoder,
        'confusion_matrix': cm.tolist()
    }
    
    with open('heartbeat_classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\nResults saved to heartbeat_classification_results.json')
    print('Best model saved to best_heartbeat_classifier.pth')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading dataset...')
    audio_paths, labels, subjects = load_dataset_from_directory(args.data_dir)
    
    # Split data (80% train, 20% validation) with stratification
    from sklearn.model_selection import train_test_split
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        audio_paths, labels, 
        test_size=0.2, 
        stratify=labels,  # Ensure balanced split
        random_state=42
    )
    
    print(f'Training samples: {len(train_paths)} (Normal: {train_labels.count(0)}, Abnormal: {train_labels.count(1)})')
    print(f'Validation samples: {len(val_paths)} (Normal: {val_labels.count(0)}, Abnormal: {val_labels.count(1)})')
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_paths, train_labels, val_paths, val_labels,
        batch_size=args.batch_size, sample_rate=args.sample_rate, 
        duration=args.duration
    )
    
    # Load best model and evaluate
    #model.load_state_dict(torch.load('best_linear_probe.pth'))
    accuracy, report, cm = evaluate_model(model, val_features, val_labels_tensor, device)
    
    print(f'\nBest Validation Accuracy: {best_val_acc:.4f}')
    print(f'Final Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(report)
    print('\nConfusion Matrix:')
    print(cm)
    
    # Save results
    results = {
        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': accuracy,
        'num_epochs': args.epochs,
        'learning_rate': args.lr
    }
    
    with open('linear_probe_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\nResults saved to linear_probe_results.json')
    print('Best model saved to best_linear_probe.pth')

if __name__ == '__main__':
    main()