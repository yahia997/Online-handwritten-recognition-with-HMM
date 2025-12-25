import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTabWidget, 
                             QTextEdit, QGroupBox, QSpinBox, QComboBox, QMessageBox,
                             QFileDialog)
from PyQt5.QtCore import Qt
import json
import os
import pickle
import pandas as pd
from model import CustomHMM # our implemented baum-welch from scratch

class DrawingCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Draw a digit (0-9) from top to bottom")
        self.ax.set_xlabel("X coordinate")
        self.ax.set_ylabel("Y coordinate")
        self.ax.grid(True)
        
        self.trajectory = []
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)
        self.points, = self.ax.plot([], [], 'ro', markersize=4)
        
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        
        self.drawing = False
        
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.drawing = True
        self.trajectory = [(event.xdata, event.ydata)]
        self.update_plot()
        
    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        if event.xdata is not None and event.ydata is not None:
            self.trajectory.append((event.xdata, event.ydata))
            self.update_plot()
        
    def on_release(self, event):
        self.drawing = False
        
    def update_plot(self):
        if self.trajectory:
            x_vals = [point[0] for point in self.trajectory]
            y_vals = [point[1] for point in self.trajectory]
            self.line.set_data(x_vals, y_vals)
            self.points.set_data(x_vals, y_vals)
            self.draw()
            
    def clear_plot(self):
        self.trajectory = []
        self.line.set_data([], [])
        self.points.set_data([], [])
        self.draw()
        
    def get_trajectory(self):
        return self.trajectory

class SimpleDigitHMM:
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.models = {}  # one HMM model per digit, will take the max likelyhood at the end
        
    def extract_features(self, trajectory, n_points=50):
        """
        Extract features using both horizontal and vertical distances from center
        """
        if len(trajectory) < 2:
            return None
            
        # convert to numpy array
        trajectory = np.array(trajectory)
        
        # find center point vertically and horizontally as we have two reference lines
        center_x = np.mean(trajectory[:, 0])
        center_y = np.mean(trajectory[:, 1])
        
        # resample to fixed number of points
        if len(trajectory) > n_points:
            # sort by Y coordinate (top to bottom)
            sorted_indices = np.argsort(trajectory[:, 1])[::-1]
            trajectory = trajectory[sorted_indices]
            
            indices = np.linspace(0, len(trajectory)-1, n_points).astype(int)
            trajectory = trajectory[indices]
        else:
            # sort by Y coordinate (top to bottom)
            sorted_indices = np.argsort(trajectory[:, 1])[::-1]
            trajectory = trajectory[sorted_indices]
            
            if len(trajectory) > 1:
                # interpolate to get fixed number of points
                from scipy.interpolate import interp1d
                t_old = np.linspace(0, 1, len(trajectory))
                t_new = np.linspace(0, 1, n_points)
                
                f_x = interp1d(t_old, trajectory[:, 0], kind='linear')
                f_y = interp1d(t_old, trajectory[:, 1], kind='linear')
                
                x_new = f_x(t_new)
                y_new = f_y(t_new)
                trajectory = np.column_stack([x_new, y_new])
        
        # extract both horizontal and vertical distances
        features = []
        for point in trajectory:
            x, y = point
            dx = x - center_x  # horizontal distance
            dy = y - center_y  # vertical distance
            

            # Discretize both dx and dy, then combine
            norm_dx = (dx - np.min(trajectory[:, 0] - center_x)) / \
             (np.max(trajectory[:, 0] - center_x) - np.min(trajectory[:, 0] - center_x) + 1e-8)
            discrete_dx = int(norm_dx * 3)  # 0-3
    
            norm_dy = (dy - np.min(trajectory[:, 1] - center_y)) / \
             (np.max(trajectory[:, 1] - center_y) - np.min(trajectory[:, 1] - center_y) + 1e-8)
            discrete_dy = int(norm_dy * 3)  # 0-3
    
            # 
            combined_feature = discrete_dx * 4 + discrete_dy 

            features.append(combined_feature)
        
        return features
    
    def train(self, training_data):
        """Train one HMM for each digit using custom Baum-Welch"""
        self.models = {}
        
        for digit in range(10):
            # Get all trajectories for this digit
            digit_trajectories = [traj for d, traj in training_data if d == digit]
            
            if not digit_trajectories:
                print(f"No training data for digit {digit}")
                continue
                
            # Extract features for all samples of this digit
            sequences = []
            
            for trajectory in digit_trajectories:
                features = self.extract_features(trajectory)
                if features is not None and len(features) > 0:
                    sequences.append(features)
            
            if not sequences:
                print(f"No valid sequences for digit {digit}")
                continue
            
            
            if sequences:
                model = CustomHMM(n_states=self.n_states, n_observations=16, n_iter=100)
                model.fit(sequences[0])
                
                if model.is_trained:
                    self.models[digit] = model
                    print(f"✓ Successfully trained model for digit {digit}")
                else:
                    print(f"✗ Training failed for digit {digit}")
    
    def predict_probabilities(self, trajectory):
        """Get probabilities for all digits 0-9"""
        features = self.extract_features(trajectory)
        if features is None or len(features) == 0:
            return {digit: 0.0 for digit in range(10)}
        
        scores = {}
        
        for digit in range(10):
            if digit in self.models:
                try:
                    score = self.models[digit].score(features)
                    scores[digit] = score
                except:
                    scores[digit] = -float('inf')
            else:
                scores[digit] = -float('inf')
        
        # Convert to probabilities (softmax)
        score_values = np.array([scores[d] for d in range(10)])
        
        # Handle -inf values
        max_score = np.max(score_values)
        if max_score == -float('inf'):
            return {digit: 0.0 for digit in range(10)}
            
        # Apply softmax
        exp_scores = np.exp(score_values - max_score)
        probabilities = exp_scores / np.sum(exp_scores)
        
        return {digit: float(probabilities[i]) for i, digit in enumerate(range(10))}
    
    def save_model(self, filename):
        """Save all trained models to a file"""
        save_data = {
            'n_states': self.n_states,
            'models': {}
        }
        
        for digit, model in self.models.items():
            if model.is_trained:
                save_data['models'][digit] = {
                    'A': model.A.tolist(),
                    'B': model.B.tolist(),
                    'pi': model.pi.tolist()
                }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    def load_model(self, filename):
        """Load all trained models from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.n_states = data['n_states']
        self.models = {}
        
        for digit, model_data in data['models'].items():
            model = CustomHMM(n_states=self.n_states)
            model.A = np.array(model_data['A'])
            model.B = np.array(model_data['B'])
            model.pi = np.array(model_data['pi'])
            model.is_trained = True
            self.models[digit] = model
        
        return True

# The rest of your GUI code remains the same (TrainingTab, TestingTab, HandwrittenDigitApp, main)
# [Include all the GUI classes from your original code here - they don't need changes]

class TrainingTab(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.training_data = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Drawing area
        drawing_group = QGroupBox("Drawing Area")
        drawing_layout = QVBoxLayout()
        
        self.drawing_canvas = DrawingCanvas(self, width=6, height=5, dpi=80)
        drawing_layout.addWidget(self.drawing_canvas)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.digit_selector = QComboBox()
        self.digit_selector.addItems([str(i) for i in range(10)])
        controls_layout.addWidget(QLabel("Digit:"))
        controls_layout.addWidget(self.digit_selector)
        
        self.add_sample_btn = QPushButton("Add Training Sample")
        self.add_sample_btn.clicked.connect(self.add_training_sample)
        controls_layout.addWidget(self.add_sample_btn)
        
        self.clear_drawing_btn = QPushButton("Clear Drawing")
        self.clear_drawing_btn.clicked.connect(self.drawing_canvas.clear_plot)
        controls_layout.addWidget(self.clear_drawing_btn)
        
        drawing_layout.addLayout(controls_layout)
        drawing_group.setLayout(drawing_layout)
        layout.addWidget(drawing_group)
        
        # Training info and controls
        info_group = QGroupBox("Training Information")
        info_layout = QVBoxLayout()
        
        self.training_info = QTextEdit()
        self.training_info.setMaximumHeight(150)
        self.training_info.setReadOnly(True)
        info_layout.addWidget(QLabel("Training Log:"))
        info_layout.addWidget(self.training_info)
        
        train_controls_layout = QHBoxLayout()
        
        self.n_states_spin = QSpinBox()
        self.n_states_spin.setRange(3, 10)
        self.n_states_spin.setValue(5)
        self.n_states_spin.setPrefix("States: ")
        
        self.train_btn = QPushButton("Train Models")
        self.train_btn.clicked.connect(self.train_models)
        
        self.clear_data_btn = QPushButton("Clear All Training Data")
        self.clear_data_btn.clicked.connect(self.clear_training_data)
        
        train_controls_layout.addWidget(self.n_states_spin)
        train_controls_layout.addWidget(self.train_btn)
        train_controls_layout.addWidget(self.clear_data_btn)
        
        info_layout.addLayout(train_controls_layout)
        
        # Save/Load controls
        save_load_layout = QHBoxLayout()
        
        self.save_model_btn = QPushButton("Save Trained Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        
        self.load_model_btn = QPushButton("Load Trained Model")
        self.load_model_btn.clicked.connect(self.load_model)
        
        self.save_data_btn = QPushButton("Save Training Data")
        self.save_data_btn.clicked.connect(self.save_training_data)
        
        self.load_data_btn = QPushButton("Load Training Data")
        self.load_data_btn.clicked.connect(self.load_training_data)
        
        save_load_layout.addWidget(self.save_model_btn)
        save_load_layout.addWidget(self.load_model_btn)
        save_load_layout.addWidget(self.save_data_btn)
        save_load_layout.addWidget(self.load_data_btn)
        
        info_layout.addLayout(save_load_layout)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        self.setLayout(layout)
        self.update_training_info()
        
    def add_training_sample(self):
        trajectory = self.drawing_canvas.get_trajectory()
        if len(trajectory) < 5:
            QMessageBox.warning(self, "Warning", "Please draw a longer trajectory!")
            return
            
        digit = int(self.digit_selector.currentText())
        self.training_data.append((digit, trajectory))
        self.drawing_canvas.clear_plot()
        self.update_training_info()
        
    def update_training_info(self):
        digit_counts = {i: 0 for i in range(10)}
        for digit, _ in self.training_data:
            digit_counts[digit] += 1
            
        info_text = f"Total training samples: {len(self.training_data)}\n\n"
        info_text += "Samples per digit:\n"
        for digit in range(10):
            info_text += f"  Digit {digit}: {digit_counts[digit]} samples\n"
            
        self.training_info.setText(info_text)
        
    def train_models(self):
        if len(self.training_data) < 10:
            QMessageBox.warning(self, "Warning", "Please add at least 10 training samples!")
            return
            
        self.main_app.hmm_model = SimpleDigitHMM(n_states=self.n_states_spin.value())
        
        # Show training progress
        self.training_info.append("\nTraining started...")
        QApplication.processEvents()
        
        self.main_app.hmm_model.train(self.training_data)
        
        # Count trained models
        trained_count = len(self.main_app.hmm_model.models)
        self.training_info.append(f"Training completed! {trained_count}/10 models trained.")
        
        # Enable save model button
        self.save_model_btn.setEnabled(True)
        
        QMessageBox.information(self, "Success", f"Training completed! {trained_count}/10 models trained.")
        
    def clear_training_data(self):
        self.training_data = []
        self.update_training_info()
        
    def save_model(self):
        if not hasattr(self.main_app, 'hmm_model') or not self.main_app.hmm_model.models:
            QMessageBox.warning(self, "Warning", "No trained model to save!")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "Model Files (*.pkl)"
        )
        
        if filename:
            self.main_app.hmm_model.save_model(filename)
            self.training_info.append(f"Model saved to: {filename}")
            QMessageBox.information(self, "Success", f"Model saved to: {filename}")
    
    def load_model(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Model Files (*.pkl)"
        )
        
        if filename:
            self.main_app.hmm_model = SimpleDigitHMM()
            success = self.main_app.hmm_model.load_model(filename)
            
            if success:
                trained_count = len(self.main_app.hmm_model.models)
                self.training_info.append(f"Model loaded! {trained_count}/10 models available.")
                self.save_model_btn.setEnabled(True)
                QMessageBox.information(self, "Success", f"Model loaded! {trained_count}/10 models available.")
            else:
                QMessageBox.warning(self, "Error", "Failed to load model!")
    
    def save_training_data(self):
        if not self.training_data:
            QMessageBox.warning(self, "Warning", "No training data to save!")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Training Data", "", "JSON Files (*.json)"
        )
        
        if filename:
            # Convert trajectories to serializable format
            serializable_data = []
            for digit, trajectory in self.training_data:
                trajectory_list = [[float(x), float(y)] for x, y in trajectory]
                serializable_data.append({
                    'digit': int(digit),
                    'trajectory': trajectory_list
                })
            
            with open(filename, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            self.training_info.append(f"Training data saved to: {filename}")
            QMessageBox.information(self, "Success", f"Training data saved to: {filename}")
    
    def load_training_data(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Training Data", "", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                self.training_data = []
                for item in data:
                    digit = item['digit']
                    trajectory = [tuple(point) for point in item['trajectory']]
                    self.training_data.append((digit, trajectory))
                
                self.update_training_info()
                self.training_info.append(f"Training data loaded from: {filename}")
                QMessageBox.information(self, "Success", f"Training data loaded! {len(self.training_data)} samples.")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load training data: {e}")

class TestingTab(QWidget):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Drawing area
        drawing_group = QGroupBox("Test Drawing Area")
        drawing_layout = QVBoxLayout()
        
        self.drawing_canvas = DrawingCanvas(self, width=6, height=5, dpi=80)
        drawing_layout.addWidget(self.drawing_canvas)
        
        # Test controls
        test_controls_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("Test Drawing")
        self.test_btn.clicked.connect(self.test_drawing)
        
        self.clear_btn = QPushButton("Clear Drawing")
        self.clear_btn.clicked.connect(self.drawing_canvas.clear_plot)
        
        test_controls_layout.addWidget(self.test_btn)
        test_controls_layout.addWidget(self.clear_btn)
        
        drawing_layout.addLayout(test_controls_layout)
        drawing_group.setLayout(drawing_layout)
        layout.addWidget(drawing_group)
        
        # Results display
        results_group = QGroupBox("Recognition Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        # Probability visualization
        self.probability_canvas = FigureCanvas(Figure(figsize=(8, 3)))
        self.probability_ax = self.probability_canvas.figure.add_subplot(111)
        results_layout.addWidget(self.probability_canvas)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.setLayout(layout)
        
    def test_drawing(self):
        if not hasattr(self.main_app, 'hmm_model') or self.main_app.hmm_model is None or not self.main_app.hmm_model.models:
            QMessageBox.warning(self, "Warning", "Please train or load models first!")
            return
            
        trajectory = self.drawing_canvas.get_trajectory()
        if len(trajectory) < 5:
            QMessageBox.warning(self, "Warning", "Please draw a digit first!")
            return
            
        probabilities = self.main_app.hmm_model.predict_probabilities(trajectory)
        
        # Display results
        self.display_results(probabilities)
        
    def display_results(self, probabilities):
        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        results_text = "Recognition Results:\n\n"
        results_text += f"{'Digit':<6} {'Probability':<12} {'Confidence':<12}\n"
        results_text += "-" * 40 + "\n"
        
        for digit, prob in sorted_probs:
            confidence = "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.1 else "LOW"
            results_text += f"{digit:<6} {prob:<12.4f} {confidence:<12}\n"
        
        predicted_digit = sorted_probs[0][0]
        results_text += f"\nPredicted Digit: {predicted_digit}"
        
        self.results_text.setText(results_text)
        
        # Plot probabilities
        self.plot_probabilities(probabilities)
        
    def plot_probabilities(self, probabilities):
        self.probability_ax.clear()
        
        digits = list(range(10))
        probs = [probabilities[d] for d in digits]
        
        colors = ['red' if p == max(probs) else 'blue' for p in probs]
        
        bars = self.probability_ax.bar(digits, probs, color=colors, alpha=0.7)
        self.probability_ax.set_xlabel('Digit')
        self.probability_ax.set_ylabel('Probability')
        self.probability_ax.set_title('Recognition Probabilities for Each Digit')
        self.probability_ax.set_xticks(digits)
        self.probability_ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            if height > 0.01:  # Only label if probability is significant
                self.probability_ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{prob:.3f}', ha='center', va='bottom')
        
        self.probability_canvas.draw()

class HandwrittenDigitApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.hmm_model = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Handwritten Digit Recognition using Custom HMM (Baum-Welch)")
        self.setGeometry(100, 100, 1000, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        self.tabs = QTabWidget()
        
        self.training_tab = TrainingTab(self)
        self.testing_tab = TestingTab(self)
        
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.testing_tab, "Testing")
        
        layout.addWidget(self.tabs)
        
        self.statusBar().showMessage("Ready - Draw digits from top to bottom")

def main():
    app = QApplication(sys.argv)
    window = HandwrittenDigitApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()