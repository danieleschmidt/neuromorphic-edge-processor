"""Accuracy and performance benchmarking for neuromorphic models."""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from ..benchmarks.performance_benchmarks import BenchmarkResult


@dataclass 
class AccuracyMetrics:
    """Container for accuracy-related metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Optional[np.ndarray] = None
    class_accuracies: Optional[Dict[int, float]] = None


class AccuracyBenchmark:
    """Comprehensive accuracy benchmarking suite."""
    
    def __init__(self):
        self.results = []
    
    def evaluate_classification(
        self,
        model: torch.nn.Module,
        test_data: List[Tuple[torch.Tensor, torch.Tensor]],
        model_name: str,
        num_classes: int
    ) -> BenchmarkResult:
        """Evaluate classification accuracy."""
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_data:
                outputs = model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take first element if tuple
                
                # Convert to predictions
                if outputs.dim() > 1:
                    if outputs.shape[-1] == 1:  # Binary classification
                        predictions = (outputs > 0.5).long().squeeze()
                    else:  # Multi-class
                        predictions = torch.argmax(outputs, dim=-1)
                else:
                    predictions = outputs.round().long()
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        # Per-class accuracies
        class_accuracies = {}
        for i in range(num_classes):
            class_mask = np.array(all_targets) == i
            if class_mask.sum() > 0:
                class_acc = accuracy_score(
                    np.array(all_targets)[class_mask],
                    np.array(all_predictions)[class_mask]
                )
                class_accuracies[i] = class_acc
        
        result = BenchmarkResult(
            model_name=model_name,
            task_name="classification_accuracy",
            execution_time=0,
            throughput=0,
            memory_usage=0,
            accuracy=accuracy,
            additional_metrics={
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": conf_matrix.tolist(),
                "class_accuracies": class_accuracies,
                "num_samples": len(all_targets),
                "num_classes": num_classes
            }
        )
        
        self.results.append(result)
        return result
    
    def evaluate_temporal_sequence_prediction(
        self,
        model: torch.nn.Module,
        test_sequences: List[Tuple[torch.Tensor, torch.Tensor]],
        model_name: str,
        prediction_horizon: int = 1
    ) -> BenchmarkResult:
        """Evaluate temporal sequence prediction accuracy."""
        
        model.eval()
        all_mse = []
        all_mae = []
        temporal_accuracies = []
        
        with torch.no_grad():
            for input_seq, target_seq in test_sequences:
                outputs = model(input_seq)
                
                # Handle different output dimensions
                if outputs.dim() == 3:  # [batch, features, time]
                    predictions = outputs[:, :, -prediction_horizon:]
                    targets = target_seq[:, :, -prediction_horizon:]
                else:  # [batch, features]
                    predictions = outputs
                    targets = target_seq
                
                # Compute MSE and MAE
                mse = torch.nn.functional.mse_loss(predictions, targets).item()
                mae = torch.nn.functional.l1_loss(predictions, targets).item()
                
                all_mse.append(mse)
                all_mae.append(mae)
                
                # Temporal accuracy (correlation-based)
                if predictions.numel() > 1 and targets.numel() > 1:
                    pred_flat = predictions.flatten()
                    target_flat = targets.flatten()
                    
                    if pred_flat.var() > 1e-8 and target_flat.var() > 1e-8:
                        corr = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                        if not torch.isnan(corr):
                            temporal_accuracies.append(corr.item())
        
        # Aggregate metrics
        mean_mse = np.mean(all_mse)
        mean_mae = np.mean(all_mae)
        mean_correlation = np.mean(temporal_accuracies) if temporal_accuracies else 0.0
        
        # Normalized accuracy (1 / (1 + MSE))
        normalized_accuracy = 1.0 / (1.0 + mean_mse)
        
        result = BenchmarkResult(
            model_name=model_name,
            task_name="temporal_prediction",
            execution_time=0,
            throughput=0,
            memory_usage=0,
            accuracy=normalized_accuracy,
            additional_metrics={
                "mse": mean_mse,
                "mae": mean_mae,
                "temporal_correlation": mean_correlation,
                "prediction_horizon": prediction_horizon,
                "num_sequences": len(test_sequences),
                "rmse": np.sqrt(mean_mse)
            }
        )
        
        self.results.append(result)
        return result
    
    def evaluate_pattern_recognition(
        self,
        model: torch.nn.Module,
        test_patterns: List[Tuple[torch.Tensor, torch.Tensor]],
        model_name: str,
        pattern_types: List[str]
    ) -> BenchmarkResult:
        """Evaluate pattern recognition capabilities."""
        
        model.eval()
        pattern_accuracies = {}
        overall_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, (input_pattern, expected_output) in enumerate(test_patterns):
                pattern_type = pattern_types[i % len(pattern_types)]
                
                outputs = model(input_pattern)
                
                # Pattern-specific accuracy computation
                if pattern_type == "spike_pattern":
                    # For spike patterns, check if output spikes match expected timing
                    if hasattr(model, 'forward') and 'return_spikes' in model.forward.__code__.co_varnames:
                        _, spike_trains = model(input_pattern, return_spikes=True)
                        accuracy = self._compute_spike_pattern_similarity(spike_trains[-1], expected_output)
                    else:
                        # Fallback to MSE-based accuracy
                        mse = torch.nn.functional.mse_loss(outputs, expected_output)
                        accuracy = 1.0 / (1.0 + mse.item())
                
                elif pattern_type == "temporal_xor":
                    # XOR pattern recognition
                    predicted = (outputs > 0.5).float()
                    accuracy = (predicted == expected_output).float().mean().item()
                
                elif pattern_type == "sequence_completion":
                    # Sequence completion task
                    similarity = torch.nn.functional.cosine_similarity(
                        outputs.flatten(), expected_output.flatten(), dim=0
                    ).item()
                    accuracy = max(0, similarity)  # Clamp negative similarities to 0
                
                else:
                    # Default: correlation-based accuracy
                    if outputs.numel() == expected_output.numel():
                        corr = torch.corrcoef(torch.stack([outputs.flatten(), expected_output.flatten()]))[0, 1]
                        accuracy = corr.item() if not torch.isnan(corr) else 0.0
                    else:
                        accuracy = 0.0
                
                if pattern_type not in pattern_accuracies:
                    pattern_accuracies[pattern_type] = []
                pattern_accuracies[pattern_type].append(accuracy)
                
                overall_correct += accuracy
                total_samples += 1
        
        # Compute mean accuracies per pattern type
        mean_pattern_accuracies = {
            pattern: np.mean(accs) for pattern, accs in pattern_accuracies.items()
        }
        
        overall_accuracy = overall_correct / total_samples if total_samples > 0 else 0.0
        
        result = BenchmarkResult(
            model_name=model_name,
            task_name="pattern_recognition",
            execution_time=0,
            throughput=0,
            memory_usage=0,
            accuracy=overall_accuracy,
            additional_metrics={
                "pattern_accuracies": mean_pattern_accuracies,
                "pattern_types": pattern_types,
                "total_patterns": total_samples,
                "accuracy_std": np.std([acc for accs in pattern_accuracies.values() for acc in accs])
            }
        )
        
        self.results.append(result)
        return result
    
    def _compute_spike_pattern_similarity(
        self, 
        predicted_spikes: torch.Tensor, 
        target_spikes: torch.Tensor
    ) -> float:
        """Compute similarity between spike patterns."""
        
        if predicted_spikes.shape != target_spikes.shape:
            return 0.0
        
        # Victor-Purpura spike distance (simplified)
        # Count matching and mismatched spikes
        pred_times = torch.where(predicted_spikes.flatten() > 0.5)[0]
        target_times = torch.where(target_spikes.flatten() > 0.5)[0]
        
        if len(pred_times) == 0 and len(target_times) == 0:
            return 1.0  # Both silent
        
        if len(pred_times) == 0 or len(target_times) == 0:
            return 0.0  # One silent, one active
        
        # Simple matching: count spikes within tolerance
        tolerance = 2  # time steps
        matches = 0
        
        for pred_time in pred_times:
            min_distance = min(abs(pred_time - target_time) for target_time in target_times)
            if min_distance <= tolerance:
                matches += 1
        
        # Similarity based on precision and recall
        precision = matches / len(pred_times)
        recall = matches / len(target_times)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            return f1
        else:
            return 0.0
    
    def evaluate_robustness(
        self,
        model: torch.nn.Module,
        clean_test_data: List[Tuple[torch.Tensor, torch.Tensor]],
        noise_levels: List[float],
        model_name: str
    ) -> List[BenchmarkResult]:
        """Evaluate model robustness to noise."""
        
        results = []
        
        # Baseline performance on clean data
        clean_result = self.evaluate_classification(model, clean_test_data, f"{model_name}_clean", 2)
        clean_accuracy = clean_result.accuracy
        
        for noise_level in noise_levels:
            # Add noise to test data
            noisy_test_data = []
            for inputs, targets in clean_test_data:
                noise = torch.randn_like(inputs) * noise_level
                noisy_inputs = inputs + noise
                noisy_test_data.append((noisy_inputs, targets))
            
            # Evaluate on noisy data
            noisy_result = self.evaluate_classification(
                model, noisy_test_data, f"{model_name}_noise_{noise_level}", 2
            )
            
            # Robustness metric
            accuracy_retention = noisy_result.accuracy / clean_accuracy if clean_accuracy > 0 else 0
            
            result = BenchmarkResult(
                model_name=model_name,
                task_name="robustness_evaluation",
                execution_time=0,
                throughput=0,
                memory_usage=0,
                accuracy=noisy_result.accuracy,
                additional_metrics={
                    "noise_level": noise_level,
                    "clean_accuracy": clean_accuracy,
                    "noisy_accuracy": noisy_result.accuracy,
                    "accuracy_retention": accuracy_retention,
                    "robustness_score": 1.0 - abs(clean_accuracy - noisy_result.accuracy)
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def evaluate_few_shot_learning(
        self,
        model: torch.nn.Module,
        support_data: List[Tuple[torch.Tensor, torch.Tensor]],
        query_data: List[Tuple[torch.Tensor, torch.Tensor]],
        model_name: str,
        n_shots: int
    ) -> BenchmarkResult:
        """Evaluate few-shot learning capabilities."""
        
        # Simple few-shot evaluation using nearest neighbor in feature space
        model.eval()
        
        # Extract features from support set
        support_features = []
        support_labels = []
        
        with torch.no_grad():
            for inputs, labels in support_data[:n_shots]:  # Use only n_shots examples
                # Get model representation (assume model has feature extraction)
                if hasattr(model, 'layers'):
                    # For spiking networks, use final layer state as features
                    _, spike_trains = model(inputs, return_spikes=True)
                    features = spike_trains[-1].mean(dim=-1)  # Average over time
                else:
                    # For other models, use final output as features
                    features = model(inputs)
                
                support_features.append(features)
                support_labels.append(labels)
        
        support_features = torch.cat(support_features, dim=0)
        support_labels = torch.cat(support_labels, dim=0)
        
        # Classify query examples
        correct_predictions = 0
        total_queries = 0
        
        with torch.no_grad():
            for query_inputs, query_labels in query_data:
                # Extract query features
                if hasattr(model, 'layers'):
                    _, spike_trains = model(query_inputs, return_spikes=True)
                    query_features = spike_trains[-1].mean(dim=-1)
                else:
                    query_features = model(query_inputs)
                
                # Nearest neighbor classification
                for i in range(query_features.shape[0]):
                    query_feat = query_features[i:i+1]
                    
                    # Compute distances to support examples
                    distances = torch.cdist(query_feat, support_features)
                    nearest_idx = torch.argmin(distances)
                    
                    predicted_label = support_labels[nearest_idx]
                    true_label = query_labels[i]
                    
                    if predicted_label == true_label:
                        correct_predictions += 1
                    total_queries += 1
        
        accuracy = correct_predictions / total_queries if total_queries > 0 else 0.0
        
        result = BenchmarkResult(
            model_name=model_name,
            task_name="few_shot_learning",
            execution_time=0,
            throughput=0,
            memory_usage=0,
            accuracy=accuracy,
            additional_metrics={
                "n_shots": n_shots,
                "support_size": len(support_data),
                "query_size": len(query_data),
                "total_queries": total_queries,
                "correct_predictions": correct_predictions
            }
        )
        
        self.results.append(result)
        return result
    
    def evaluate_continual_learning(
        self,
        model: torch.nn.Module,
        task_datasets: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        model_name: str,
        learning_rate: float = 0.001
    ) -> List[BenchmarkResult]:
        """Evaluate continual learning capabilities."""
        
        results = []
        task_accuracies = []  # Track accuracy on all previous tasks
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for task_idx, task_data in enumerate(task_datasets):
            # Train on current task
            model.train()
            for epoch in range(10):  # Small number of epochs
                for inputs, targets in task_data[:50]:  # Limit training data
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    if outputs.dim() > 1 and outputs.shape[-1] > 1:
                        loss = torch.nn.functional.cross_entropy(outputs, targets.long())
                    else:
                        loss = torch.nn.functional.mse_loss(outputs.squeeze(), targets.float())
                    
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on all tasks seen so far
            current_task_accuracies = []
            
            for eval_task_idx in range(task_idx + 1):
                eval_data = task_datasets[eval_task_idx][-20:]  # Use test portion
                
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, targets in eval_data:
                        outputs = model(inputs)
                        
                        if outputs.dim() > 1 and outputs.shape[-1] > 1:
                            predictions = torch.argmax(outputs, dim=-1)
                            correct += (predictions == targets).sum().item()
                            total += targets.numel()
                        else:
                            # Regression-like task
                            mse = torch.nn.functional.mse_loss(outputs.squeeze(), targets.float())
                            accuracy = 1.0 / (1.0 + mse.item())
                            correct += accuracy
                            total += 1
                
                task_accuracy = correct / total if total > 0 else 0.0
                current_task_accuracies.append(task_accuracy)
            
            task_accuracies.append(current_task_accuracies)
            
            # Compute forgetting metric
            if task_idx > 0:
                forgetting = 0.0
                for prev_task_idx in range(task_idx):
                    # Compare current accuracy to accuracy right after learning that task
                    original_accuracy = task_accuracies[prev_task_idx][prev_task_idx]
                    current_accuracy = current_task_accuracies[prev_task_idx]
                    forgetting += max(0, original_accuracy - current_accuracy)
                
                average_forgetting = forgetting / task_idx
            else:
                average_forgetting = 0.0
            
            # Current task accuracy
            current_accuracy = current_task_accuracies[-1]
            
            result = BenchmarkResult(
                model_name=model_name,
                task_name="continual_learning",
                execution_time=0,
                throughput=0,
                memory_usage=0,
                accuracy=current_accuracy,
                additional_metrics={
                    "task_number": task_idx + 1,
                    "average_forgetting": average_forgetting,
                    "all_task_accuracies": current_task_accuracies,
                    "backward_transfer": -average_forgetting,  # Negative forgetting
                    "forward_transfer": 0.0  # Would need to compute properly
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def generate_accuracy_report(self) -> Dict:
        """Generate comprehensive accuracy report."""
        
        report = {
            "summary": {
                "total_evaluations": len(self.results),
                "evaluation_tasks": list(set(r.task_name for r in self.results)),
                "models_evaluated": list(set(r.model_name for r in self.results))
            },
            "accuracy_rankings": {},
            "task_analysis": {}
        }
        
        # Group by task and rank models
        tasks = set(r.task_name for r in self.results)
        
        for task in tasks:
            task_results = [r for r in self.results if r.task_name == task]
            
            # Rank by accuracy
            accuracy_ranking = sorted(
                task_results,
                key=lambda x: x.accuracy if x.accuracy is not None else 0,
                reverse=True
            )
            
            report["accuracy_rankings"][task] = [
                {
                    "model": r.model_name,
                    "accuracy": r.accuracy,
                    "additional_metrics": r.additional_metrics
                }
                for r in accuracy_ranking[:5]  # Top 5
            ]
            
            # Task-specific analysis
            accuracies = [r.accuracy for r in task_results if r.accuracy is not None]
            if accuracies:
                report["task_analysis"][task] = {
                    "mean_accuracy": np.mean(accuracies),
                    "std_accuracy": np.std(accuracies),
                    "best_accuracy": max(accuracies),
                    "worst_accuracy": min(accuracies),
                    "num_models": len(accuracies)
                }
        
        return report
    
    def save_accuracy_results(self, filename: str = "accuracy_benchmark_results.json"):
        """Save accuracy benchmark results to file."""
        import json
        
        report = self.generate_accuracy_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Accuracy benchmark results saved to {filename}")