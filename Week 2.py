# Plant Disease Detection - Prediction & Evaluation Module
# Additional 40% of complete project

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import cv2
import os
import json
from PIL import Image

# =====================================
# 1. PREDICTION & INFERENCE MODULE
# =====================================

class DiseasePredictor:
    def __init__(self, model_path, class_names=None):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model (.h5 file)
            class_names: List of class names in order
        """
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = (224, 224)
        
    def load_class_names(self, class_names_path):
        """Load class names from JSON file"""
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
    
    def preprocess_image(self, img_path):
        """
        Preprocess single image for prediction
        
        Args:
            img_path: Path to image file or numpy array
        """
        if isinstance(img_path, str):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_path
            
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict_single(self, img_path, top_k=3):
        """
        Predict disease for single image
        
        Args:
            img_path: Path to image
            top_k: Return top K predictions
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        img = self.preprocess_image(img_path)
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Get top K predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = {
            'predictions': [],
            'image_path': img_path if isinstance(img_path, str) else 'array'
        }
        
        for idx in top_indices:
            class_name = self.class_names[idx] if self.class_names else f"Class_{idx}"
            confidence = float(predictions[idx]) * 100
            results['predictions'].append({
                'class': class_name,
                'confidence': confidence
            })
        
        return results
    
    def predict_batch(self, img_paths, batch_size=32):
        """
        Predict diseases for multiple images
        
        Args:
            img_paths: List of image paths
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i+batch_size]
            batch_imgs = np.vstack([self.preprocess_image(p) for p in batch_paths])
            
            predictions = self.model.predict(batch_imgs, verbose=0)
            
            for j, pred in enumerate(predictions):
                top_idx = np.argmax(pred)
                class_name = self.class_names[top_idx] if self.class_names else f"Class_{top_idx}"
                confidence = float(pred[top_idx]) * 100
                
                results.append({
                    'image_path': batch_paths[j],
                    'predicted_class': class_name,
                    'confidence': confidence
                })
        
        return results
    
    def predict_with_threshold(self, img_path, confidence_threshold=0.7):
        """
        Predict with confidence threshold
        
        Args:
            img_path: Path to image
            confidence_threshold: Minimum confidence (0-1)
            
        Returns:
            Prediction result or 'uncertain' flag
        """
        result = self.predict_single(img_path, top_k=1)
        top_prediction = result['predictions'][0]
        
        if top_prediction['confidence'] / 100 >= confidence_threshold:
            return {
                'status': 'confident',
                'prediction': top_prediction
            }
        else:
            return {
                'status': 'uncertain',
                'prediction': top_prediction,
                'message': 'Low confidence. Please consult an expert.'
            }
    
    def visualize_prediction(self, img_path, save_path=None):
        """Visualize prediction with image"""
        result = self.predict_single(img_path, top_k=3)
        
        # Load and display image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 5))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input Image')
        
        # Display predictions
        plt.subplot(1, 2, 2)
        classes = [p['class'] for p in result['predictions']]
        confidences = [p['confidence'] for p in result['predictions']]
        
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(classes))]
        plt.barh(classes, confidences, color=colors)
        plt.xlabel('Confidence (%)')
        plt.title('Top 3 Predictions')
        plt.xlim(0, 100)
        
        for i, (cls, conf) in enumerate(zip(classes, confidences)):
            plt.text(conf + 1, i, f'{conf:.2f}%', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# =====================================
# 2. MODEL EVALUATION MODULE
# =====================================

class ModelEvaluator:
    def __init__(self, model, test_generator, class_names):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
            test_generator: Test data generator
            class_names: List of class names
        """
        self.model = model
        self.test_gen = test_generator
        self.class_names = class_names
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def generate_predictions(self):
        """Generate predictions on test set"""
        print("Generating predictions on test set...")
        
        # Reset generator
        self.test_gen.reset()
        
        # Get predictions
        self.y_pred_proba = self.model.predict(self.test_gen, verbose=1)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        
        # Get true labels
        self.y_true = self.test_gen.classes
        
        print(f"Predictions generated for {len(self.y_true)} samples")
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        if self.y_pred is None:
            self.generate_predictions()
        
        # Overall metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted'
        )
        
        accuracy = np.mean(self.y_pred == self.y_true)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print("\n" + "="*50)
        print("OVERALL METRICS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("="*50 + "\n")
        
        return metrics
    
    def plot_confusion_matrix(self, save_path='confusion_matrix.png', figsize=(15, 12)):
        """Plot confusion matrix"""
        if self.y_pred is None:
            self.generate_predictions()
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_normalized_confusion_matrix(self, save_path='confusion_matrix_normalized.png'):
        """Plot normalized confusion matrix (percentages)"""
        if self.y_pred is None:
            self.generate_predictions()
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage'})
        plt.title('Normalized Confusion Matrix (%)', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, save_path='classification_report.txt'):
        """Generate detailed classification report"""
        if self.y_pred is None:
            self.generate_predictions()
        
        report = classification_report(
            self.y_true, self.y_pred, 
            target_names=self.class_names,
            digits=4
        )
        
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(report)
        print("="*80 + "\n")
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(report)
        
        print(f"Classification report saved to {save_path}")
        
        return report
    
    def plot_per_class_accuracy(self, save_path='per_class_accuracy.png'):
        """Plot accuracy for each class"""
        if self.y_pred is None:
            self.generate_predictions()
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Sort by accuracy
        sorted_indices = np.argsort(per_class_acc)
        sorted_classes = [self.class_names[i] for i in sorted_indices]
        sorted_acc = per_class_acc[sorted_indices]
        
        plt.figure(figsize=(12, 10))
        colors = ['#e74c3c' if acc < 0.8 else '#f39c12' if acc < 0.9 else '#2ecc71' 
                 for acc in sorted_acc]
        
        plt.barh(sorted_classes, sorted_acc * 100, color=colors)
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=14, pad=20)
        plt.xlim(0, 100)
        
        # Add value labels
        for i, acc in enumerate(sorted_acc):
            plt.text(acc * 100 + 1, i, f'{acc*100:.1f}%', va='center')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', label='<80%'),
            Patch(facecolor='#f39c12', label='80-90%'),
            Patch(facecolor='#2ecc71', label='>90%')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Per-class accuracy plot saved to {save_path}")
    
    def plot_misclassified_samples(self, num_samples=10, save_path='misclassified_samples.png'):
        """Visualize misclassified samples"""
        if self.y_pred is None:
            self.generate_predictions()
        
        # Find misclassified indices
        misclassified_idx = np.where(self.y_pred != self.y_true)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassified samples found!")
            return
        
        # Select random samples
        sample_idx = np.random.choice(misclassified_idx, 
                                     min(num_samples, len(misclassified_idx)), 
                                     replace=False)
        
        # Plot samples
        rows = int(np.ceil(len(sample_idx) / 5))
        fig, axes = plt.subplots(rows, 5, figsize=(20, rows * 4))
        axes = axes.flatten() if rows > 1 else [axes]
        
        for i, idx in enumerate(sample_idx):
            # Get image
            img_path = self.test_gen.filepaths[idx]
            img = Image.open(img_path)
            
            # Get predictions
            true_label = self.class_names[self.y_true[idx]]
            pred_label = self.class_names[self.y_pred[idx]]
            confidence = self.y_pred_proba[idx][self.y_pred[idx]] * 100
            
            # Plot
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)',
                            fontsize=10, color='red')
        
        # Hide unused subplots
        for j in range(len(sample_idx), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        
        print(f"Misclassified samples visualization saved to {save_path}")
    
    def generate_evaluation_summary(self, output_dir='evaluation_results'):
        """Generate complete evaluation summary"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("GENERATING COMPLETE EVALUATION SUMMARY")
        print("="*80 + "\n")
        
        # Generate all evaluations
        metrics = self.calculate_metrics()
        self.plot_confusion_matrix(save_path=f'{output_dir}/confusion_matrix.png')
        self.plot_normalized_confusion_matrix(save_path=f'{output_dir}/confusion_matrix_normalized.png')
        self.generate_classification_report(save_path=f'{output_dir}/classification_report.txt')
        self.plot_per_class_accuracy(save_path=f'{output_dir}/per_class_accuracy.png')
        self.plot_misclassified_samples(save_path=f'{output_dir}/misclassified_samples.png')
        
        # Save metrics to JSON
        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n✅ All evaluation results saved to '{output_dir}/' directory")
        print("="*80 + "\n")


# =====================================
# 3. EXAMPLE USAGE
# =====================================

if __name__ == "__main__":
    # ============= PREDICTION EXAMPLE =============
    print("\n" + "="*60)
    print("PREDICTION MODULE EXAMPLE")
    print("="*60 + "\n")
    
    # Initialize predictor
    predictor = DiseasePredictor(
        model_path='models/best_model.h5',
        class_names=None  # Will load from test generator
    )
    
    # Single prediction
    img_path = 'test_images/tomato_leaf.jpg'
    result = predictor.predict_single(img_path, top_k=3)
    
    print(f"Image: {result['image_path']}")
    print("\nTop 3 Predictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"{i}. {pred['class']}: {pred['confidence']:.2f}%")
    
    # Visualize prediction
    predictor.visualize_prediction(img_path, save_path='prediction_result.png')
    
    # Prediction with threshold
    threshold_result = predictor.predict_with_threshold(img_path, confidence_threshold=0.7)
    print(f"\nPrediction Status: {threshold_result['status']}")
    
    
    # ============= EVALUATION EXAMPLE =============
    print("\n" + "="*60)
    print("EVALUATION MODULE EXAMPLE")
    print("="*60 + "\n")
    
    # Load model and test data
    model = keras.models.load_model('models/best_model.h5')
    
    # Create test generator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_generator=test_generator,
        class_names=list(test_generator.class_indices.keys())
    )
    
    # Generate complete evaluation
    evaluator.generate_evaluation_summary(output_dir='evaluation_results')
    
    print("\n✅ Prediction and Evaluation modules complete!")
    print("Check 'evaluation_results/' folder for all visualizations")
