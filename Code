# Plant Disease Detection using CNN
# Core Implementation (40% of complete project)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. DATA PREPROCESSING & AUGMENTATION
# =====================================

class DataLoader:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        
    def create_data_generators(self, train_dir, val_dir):
        """Create training and validation data generators with augmentation"""
        
        # Training data augmentation for better generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data - only rescaling
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        return train_generator, val_generator


# =====================================
# 2. CNN MODEL ARCHITECTURE
# =====================================

class PlantDiseaseClassifier:
    def __init__(self, num_classes, img_size=(224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        
    def build_custom_cnn(self):
        """Build a custom CNN from scratch"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_transfer_learning_model(self):
        """Build model using transfer learning with MobileNetV2"""
        # Load pre-trained MobileNetV2 without top layers
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
    
    def get_model_summary(self):
        """Display model architecture"""
        return self.model.summary()


# =====================================
# 3. TRAINING PIPELINE
# =====================================

class ModelTrainer:
    def __init__(self, model, train_gen, val_gen):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.history = None
        
    def setup_callbacks(self, model_save_path='best_model.h5'):
        """Setup training callbacks"""
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, epochs=50, callbacks=None):
        """Train the model"""
        if callbacks is None:
            callbacks = self.setup_callbacks()
            
        self.history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return self.history
    
    def plot_training_history(self):
        """Visualize training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()


# =====================================
# 4. EXAMPLE USAGE
# =====================================

if __name__ == "__main__":
    # Configuration
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_CLASSES = 38  # For PlantVillage dataset
    EPOCHS = 50
    
    # Paths (update with your dataset paths)
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/validation'
    
    # Step 1: Load and prepare data
    print("Loading data...")
    data_loader = DataLoader(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    train_gen, val_gen = data_loader.create_data_generators(TRAIN_DIR, VAL_DIR)
    
    print(f"Found {train_gen.samples} training images")
    print(f"Found {val_gen.samples} validation images")
    print(f"Number of classes: {len(train_gen.class_indices)}")
    
    # Step 2: Build model
    print("\nBuilding model...")
    classifier = PlantDiseaseClassifier(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    
    # Choose one: Custom CNN or Transfer Learning
    # model = classifier.build_custom_cnn()
    model = classifier.build_transfer_learning_model()
    
    classifier.compile_model(learning_rate=0.001)
    classifier.get_model_summary()
    
    # Step 3: Train model
    print("\nTraining model...")
    trainer = ModelTrainer(classifier.model, train_gen, val_gen)
    history = trainer.train(epochs=EPOCHS)
    
    # Step 4: Visualize results
    print("\nPlotting training history...")
    trainer.plot_training_history()
    
    print("\nTraining complete! Model saved as 'best_model.h5'")
