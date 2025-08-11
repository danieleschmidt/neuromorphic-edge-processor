"""Statistical analysis and validation for neuromorphic research."""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare
from scipy.stats import ttest_ind, ttest_rel, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StatisticalTest:
    """Results from a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    significant: bool
    alpha: float = 0.05


@dataclass
class ComparisonResult:
    """Results from model comparison analysis."""
    models_compared: List[str]
    metrics_compared: List[str]
    statistical_tests: List[StatisticalTest]
    effect_sizes: Dict[str, float]
    rankings: Dict[str, List[str]]
    best_model: str
    confidence_level: float
    sample_size: int


class EffectSizeCalculator:
    """Calculate various effect size measures."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size.
        
        Args:
            group1, group2: Data arrays to compare
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d
    
    @staticmethod
    def glass_delta(group1: np.ndarray, group2: np.ndarray, control_group: int = 2) -> float:
        """Calculate Glass's delta effect size.
        
        Args:
            group1, group2: Data arrays to compare
            control_group: Which group to use as control (1 or 2)
            
        Returns:
            Glass's delta effect size
        """
        if control_group == 1:
            control_std = np.std(group1, ddof=1)
        else:
            control_std = np.std(group2, ddof=1)
        
        delta = (np.mean(group1) - np.mean(group2)) / control_std
        return delta
    
    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d).
        
        Args:
            group1, group2: Data arrays to compare
            
        Returns:
            Hedges' g effect size
        """
        n1, n2 = len(group1), len(group2)
        
        # Cohen's d
        d = EffectSizeCalculator.cohens_d(group1, group2)
        
        # Correction factor
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        
        g = d * correction
        return g
    
    @staticmethod
    def cliff_delta(group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size).
        
        Args:
            group1, group2: Data arrays to compare
            
        Returns:
            Cliff's delta (-1 to 1)
        """
        n1, n2 = len(group1), len(group2)
        
        # Count pairs where group1 > group2 and group2 > group1
        greater = 0
        less = 0
        
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    greater += 1
                elif x1 < x2:
                    less += 1
        
        delta = (greater - less) / (n1 * n2)
        return delta
    
    @staticmethod
    def eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
        """Calculate eta-squared effect size for ANOVA.
        
        Args:
            f_stat: F-statistic from ANOVA
            df_between: Degrees of freedom between groups
            df_within: Degrees of freedom within groups
            
        Returns:
            Eta-squared effect size
        """
        eta_sq = (df_between * f_stat) / (df_between * f_stat + df_within)
        return eta_sq
    
    @staticmethod
    def interpret_effect_size(effect_size: float, measure_type: str) -> str:
        """Interpret effect size magnitude.
        
        Args:
            effect_size: Effect size value
            measure_type: Type of effect size measure
            
        Returns:
            Interpretation string
        """
        abs_effect = abs(effect_size)
        
        if measure_type in ["cohens_d", "hedges_g", "glass_delta"]:
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        
        elif measure_type == "cliff_delta":
            if abs_effect < 0.147:
                return "negligible"
            elif abs_effect < 0.33:
                return "small"
            elif abs_effect < 0.474:
                return "medium"
            else:
                return "large"
        
        elif measure_type == "eta_squared":
            if abs_effect < 0.01:
                return "small"
            elif abs_effect < 0.06:
                return "medium"
            else:
                return "large"
        
        else:
            return "unknown"


class StatisticalAnalyzer:
    """Comprehensive statistical analysis for neuromorphic research validation."""
    
    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for statistical tests
            random_seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.random_seed = random_seed
        self.effect_calc = EffectSizeCalculator()
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Store analysis history
        self.analysis_history: List[Dict] = []
    
    def compare_two_groups(
        self,
        group1_data: Union[List, np.ndarray],
        group2_data: Union[List, np.ndarray],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        paired: bool = False,
        assume_normality: Optional[bool] = None
    ) -> List[StatisticalTest]:
        """Compare two groups using appropriate statistical tests.
        
        Args:
            group1_data, group2_data: Data for comparison
            group1_name, group2_name: Names for the groups
            paired: Whether data is paired (repeated measures)
            assume_normality: Whether to assume normal distribution
            
        Returns:
            List of statistical test results
        """
        group1 = np.array(group1_data)
        group2 = np.array(group2_data)
        
        results = []
        
        # Test for normality if not specified
        if assume_normality is None:
            _, p1 = stats.normaltest(group1)
            _, p2 = stats.normaltest(group2)
            assume_normality = (p1 > self.alpha and p2 > self.alpha)
        
        # Choose appropriate test
        if assume_normality:
            # Parametric tests
            if paired:
                # Paired t-test
                statistic, p_value = ttest_rel(group1, group2)
                test_name = "Paired t-test"
                effect_size = self.effect_calc.cohens_d(group1, group2)
            else:
                # Independent t-test
                statistic, p_value = ttest_ind(group1, group2)
                test_name = "Independent t-test"
                effect_size = self.effect_calc.cohens_d(group1, group2)
        else:
            # Non-parametric tests
            if paired:
                # Wilcoxon signed-rank test
                statistic, p_value = wilcoxon(group1, group2)
                test_name = "Wilcoxon signed-rank test"
                effect_size = self.effect_calc.cliff_delta(group1, group2)
            else:
                # Mann-Whitney U test
                statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                test_name = "Mann-Whitney U test"
                effect_size = self.effect_calc.cliff_delta(group1, group2)
        
        # Calculate confidence interval for mean difference
        if assume_normality:
            se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
            mean_diff = np.mean(group1) - np.mean(group2)
            ci_margin = stats.t.ppf(1 - self.alpha/2, len(group1) + len(group2) - 2) * se_diff
            ci = (mean_diff - ci_margin, mean_diff + ci_margin)
        else:
            ci = None
        
        # Interpret results
        significant = p_value < self.alpha
        interpretation = self._interpret_comparison(
            group1_name, group2_name, statistic, p_value, effect_size, significant
        )
        
        test_result = StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation,
            significant=significant,
            alpha=self.alpha
        )
        
        results.append(test_result)
        
        return results
    
    def compare_multiple_groups(
        self,
        groups_data: Dict[str, Union[List, np.ndarray]],
        assume_normality: Optional[bool] = None,
        post_hoc: bool = True
    ) -> List[StatisticalTest]:
        """Compare multiple groups using ANOVA or Kruskal-Wallis test.
        
        Args:
            groups_data: Dictionary mapping group names to data arrays
            assume_normality: Whether to assume normal distribution
            post_hoc: Whether to perform post-hoc pairwise comparisons
            
        Returns:
            List of statistical test results
        """
        group_names = list(groups_data.keys())
        group_arrays = [np.array(data) for data in groups_data.values()]
        
        results = []
        
        # Test for normality if not specified
        if assume_normality is None:
            normality_tests = [stats.normaltest(arr)[1] for arr in group_arrays]
            assume_normality = all(p > self.alpha for p in normality_tests)
        
        # Overall comparison
        if assume_normality:
            # One-way ANOVA
            statistic, p_value = stats.f_oneway(*group_arrays)
            test_name = "One-way ANOVA"
            
            # Calculate eta-squared
            ss_between = sum(len(arr) * (np.mean(arr) - np.mean(np.concatenate(group_arrays)))**2 
                           for arr in group_arrays)
            ss_total = sum(np.sum((arr - np.mean(np.concatenate(group_arrays)))**2) 
                          for arr in group_arrays)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
            effect_size = eta_squared
        else:
            # Kruskal-Wallis test
            statistic, p_value = kruskal(*group_arrays)
            test_name = "Kruskal-Wallis test"
            
            # Approximate eta-squared for Kruskal-Wallis
            n_total = sum(len(arr) for arr in group_arrays)
            eta_squared = (statistic - len(group_arrays) + 1) / (n_total - len(group_arrays))
            effect_size = max(0, eta_squared)  # Ensure non-negative
        
        significant = p_value < self.alpha
        interpretation = self._interpret_multiple_groups(
            group_names, statistic, p_value, effect_size, significant, test_name
        )
        
        overall_result = StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=None,
            interpretation=interpretation,
            significant=significant,
            alpha=self.alpha
        )
        
        results.append(overall_result)
        
        # Post-hoc pairwise comparisons if significant
        if significant and post_hoc and len(group_names) > 2:
            pairwise_results = self._pairwise_comparisons(
                groups_data, assume_normality, bonferroni_correction=True
            )
            results.extend(pairwise_results)
        
        return results
    
    def _pairwise_comparisons(
        self,
        groups_data: Dict[str, Union[List, np.ndarray]],
        assume_normality: bool,
        bonferroni_correction: bool = True
    ) -> List[StatisticalTest]:
        """Perform pairwise post-hoc comparisons.
        
        Args:
            groups_data: Dictionary of group data
            assume_normality: Whether to use parametric tests
            bonferroni_correction: Whether to apply Bonferroni correction
            
        Returns:
            List of pairwise comparison results
        """
        group_names = list(groups_data.keys())
        results = []
        
        # Number of comparisons for Bonferroni correction
        num_comparisons = len(group_names) * (len(group_names) - 1) // 2
        adjusted_alpha = self.alpha / num_comparisons if bonferroni_correction else self.alpha
        
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name1, name2 = group_names[i], group_names[j]
                data1, data2 = np.array(groups_data[name1]), np.array(groups_data[name2])
                
                # Perform pairwise test
                pairwise_results = self.compare_two_groups(
                    data1, data2, name1, name2, 
                    paired=False, assume_normality=assume_normality
                )
                
                # Adjust for multiple comparisons
                for result in pairwise_results:
                    result.alpha = adjusted_alpha
                    result.significant = result.p_value < adjusted_alpha
                    result.test_name = f"Post-hoc {result.test_name}"
                    if bonferroni_correction:
                        result.test_name += " (Bonferroni corrected)"
                
                results.extend(pairwise_results)
        
        return results
    
    def validate_model_performance(
        self,
        model_results: Dict[str, Dict[str, List[float]]],
        metrics: List[str] = ["accuracy", "precision", "recall", "f1"],
        reference_model: Optional[str] = None
    ) -> ComparisonResult:
        """Validate and compare model performance across metrics.
        
        Args:
            model_results: Dict mapping model names to metric results
            metrics: List of metrics to compare
            reference_model: Reference model for comparison
            
        Returns:
            Comprehensive comparison result
        """
        model_names = list(model_results.keys())
        
        if reference_model and reference_model not in model_names:
            raise ValueError(f"Reference model '{reference_model}' not found in results")
        
        all_tests = []
        effect_sizes = {}
        rankings = {}
        
        # Compare each metric across models
        for metric in metrics:
            if not all(metric in results for results in model_results.values()):
                continue
            
            metric_data = {name: results[metric] for name, results in model_results.items()}
            
            # Overall comparison
            if len(model_names) > 2:
                metric_tests = self.compare_multiple_groups(
                    metric_data, assume_normality=None, post_hoc=True
                )
            else:
                # Two-group comparison
                names = list(metric_data.keys())
                metric_tests = self.compare_two_groups(
                    metric_data[names[0]], metric_data[names[1]], 
                    names[0], names[1], paired=False
                )
            
            all_tests.extend(metric_tests)
            
            # Calculate effect sizes between models
            if reference_model:
                ref_data = np.array(metric_data[reference_model])
                for model_name in model_names:
                    if model_name != reference_model:
                        model_data = np.array(metric_data[model_name])
                        effect_key = f"{metric}_{reference_model}_vs_{model_name}"
                        effect_sizes[effect_key] = self.effect_calc.cohens_d(model_data, ref_data)
            
            # Rank models by metric
            model_means = {name: np.mean(data) for name, data in metric_data.items()}
            ranked_models = sorted(model_means.keys(), key=lambda x: model_means[x], reverse=True)
            rankings[metric] = ranked_models
        
        # Determine best overall model
        if rankings:
            # Average rank across metrics
            model_avg_ranks = {}
            for model in model_names:
                ranks = [rankings[metric].index(model) + 1 for metric in rankings.keys()]
                model_avg_ranks[model] = np.mean(ranks)
            
            best_model = min(model_avg_ranks.keys(), key=lambda x: model_avg_ranks[x])
        else:
            best_model = model_names[0] if model_names else ""
        
        # Calculate confidence level
        significant_tests = [test for test in all_tests if test.significant]
        confidence_level = len(significant_tests) / max(1, len(all_tests))
        
        # Sample size (minimum across all models and metrics)
        sample_size = min(
            len(results[metric]) 
            for results in model_results.values() 
            for metric in metrics 
            if metric in results
        ) if model_results else 0
        
        comparison_result = ComparisonResult(
            models_compared=model_names,
            metrics_compared=metrics,
            statistical_tests=all_tests,
            effect_sizes=effect_sizes,
            rankings=rankings,
            best_model=best_model,
            confidence_level=confidence_level,
            sample_size=sample_size
        )
        
        return comparison_result
    
    def analyze_learning_convergence(
        self,
        learning_curves: Dict[str, List[float]],
        significance_threshold: float = 0.01,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """Analyze learning convergence patterns and stability.
        
        Args:
            learning_curves: Dict mapping model names to learning curves
            significance_threshold: Threshold for detecting convergence
            window_size: Window size for stability analysis
            
        Returns:
            Convergence analysis results
        """
        results = {}
        
        for model_name, curve in learning_curves.items():
            curve_array = np.array(curve)
            
            # Detect convergence point
            convergence_epoch = self._detect_convergence(curve_array, significance_threshold)
            
            # Analyze stability in final epochs
            if len(curve_array) >= window_size:
                final_window = curve_array[-window_size:]
                stability_score = 1.0 / (1.0 + np.std(final_window))
                
                # Trend analysis
                epochs = np.arange(len(curve_array))
                slope, intercept, r_value, p_value, std_err = stats.linregress(epochs, curve_array)
                
                # Monotonicity test
                monotonic = self._test_monotonicity(curve_array)
                
            else:
                stability_score = 0.0
                slope = 0.0
                r_value = 0.0
                p_value = 1.0
                monotonic = False
            
            results[model_name] = {
                "convergence_epoch": convergence_epoch,
                "converged": convergence_epoch > 0,
                "stability_score": stability_score,
                "trend_slope": slope,
                "trend_correlation": r_value,
                "trend_p_value": p_value,
                "is_monotonic": monotonic,
                "final_performance": curve_array[-1] if len(curve_array) > 0 else 0.0,
                "best_performance": np.max(curve_array) if len(curve_array) > 0 else 0.0,
                "performance_variance": np.var(curve_array) if len(curve_array) > 0 else 0.0
            }
        
        return results
    
    def _detect_convergence(self, curve: np.ndarray, threshold: float) -> int:
        """Detect convergence point in learning curve.
        
        Args:
            curve: Learning curve array
            threshold: Significance threshold for convergence
            
        Returns:
            Convergence epoch (0 if not converged)
        """
        if len(curve) < 10:
            return 0
        
        # Look for point where improvement becomes insignificant
        for i in range(10, len(curve)):
            recent_segment = curve[i-10:i]
            
            # Test if recent changes are significantly different from zero
            if len(recent_segment) > 1:
                changes = np.diff(recent_segment)
                _, p_value = stats.ttest_1samp(changes, 0)
                
                if p_value > threshold:  # No significant improvement
                    return i
        
        return 0  # No convergence detected
    
    def _test_monotonicity(self, curve: np.ndarray) -> bool:
        """Test if learning curve is monotonic.
        
        Args:
            curve: Learning curve array
            
        Returns:
            True if monotonic (generally increasing)
        """
        if len(curve) < 2:
            return True
        
        differences = np.diff(curve)
        positive_changes = np.sum(differences > 0)
        total_changes = len(differences)
        
        # Consider monotonic if at least 70% of changes are positive
        return (positive_changes / total_changes) >= 0.7
    
    def _interpret_comparison(
        self,
        group1_name: str,
        group2_name: str,
        statistic: float,
        p_value: float,
        effect_size: float,
        significant: bool
    ) -> str:
        """Generate interpretation for two-group comparison.
        
        Args:
            group1_name, group2_name: Group names
            statistic: Test statistic
            p_value: P-value
            effect_size: Effect size
            significant: Whether result is significant
            
        Returns:
            Interpretation string
        """
        effect_magnitude = self.effect_calc.interpret_effect_size(effect_size, "cohens_d")
        
        if significant:
            direction = "higher" if effect_size > 0 else "lower"
            interpretation = (
                f"{group1_name} performs significantly {direction} than {group2_name} "
                f"(p = {p_value:.4f}, effect size = {effect_size:.3f}, magnitude = {effect_magnitude})"
            )
        else:
            interpretation = (
                f"No significant difference between {group1_name} and {group2_name} "
                f"(p = {p_value:.4f}, effect size = {effect_size:.3f})"
            )
        
        return interpretation
    
    def _interpret_multiple_groups(
        self,
        group_names: List[str],
        statistic: float,
        p_value: float,
        effect_size: float,
        significant: bool,
        test_name: str
    ) -> str:
        """Generate interpretation for multiple group comparison.
        
        Args:
            group_names: List of group names
            statistic: Test statistic
            p_value: P-value
            effect_size: Effect size
            significant: Whether result is significant
            test_name: Name of statistical test
            
        Returns:
            Interpretation string
        """
        effect_magnitude = self.effect_calc.interpret_effect_size(effect_size, "eta_squared")
        
        if significant:
            interpretation = (
                f"Significant differences found among groups {group_names} using {test_name} "
                f"(p = {p_value:.4f}, η² = {effect_size:.3f}, magnitude = {effect_magnitude})"
            )
        else:
            interpretation = (
                f"No significant differences among groups {group_names} using {test_name} "
                f"(p = {p_value:.4f}, η² = {effect_size:.3f})"
            )
        
        return interpretation
    
    def generate_validation_report(
        self,
        comparison_result: ComparisonResult,
        learning_analysis: Optional[Dict[str, Any]] = None,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report.
        
        Args:
            comparison_result: Model comparison results
            learning_analysis: Learning convergence analysis
            include_plots: Whether to generate plots
            
        Returns:
            Validation report dictionary
        """
        report = {
            "executive_summary": self._generate_executive_summary(comparison_result),
            "statistical_results": self._format_statistical_results(comparison_result.statistical_tests),
            "model_rankings": comparison_result.rankings,
            "effect_sizes": comparison_result.effect_sizes,
            "recommendations": self._generate_recommendations(comparison_result),
            "methodology": {
                "significance_level": self.alpha,
                "sample_size": comparison_result.sample_size,
                "models_compared": comparison_result.models_compared,
                "metrics_evaluated": comparison_result.metrics_compared
            }
        }
        
        if learning_analysis:
            report["learning_analysis"] = learning_analysis
            report["convergence_summary"] = self._summarize_convergence(learning_analysis)
        
        if include_plots:
            report["plots"] = self._generate_validation_plots(comparison_result, learning_analysis)
        
        return report
    
    def _generate_executive_summary(self, comparison_result: ComparisonResult) -> str:
        """Generate executive summary of results.
        
        Args:
            comparison_result: Comparison results
            
        Returns:
            Executive summary string
        """
        best_model = comparison_result.best_model
        confidence = comparison_result.confidence_level
        num_models = len(comparison_result.models_compared)
        num_metrics = len(comparison_result.metrics_compared)
        
        significant_tests = sum(1 for test in comparison_result.statistical_tests if test.significant)
        total_tests = len(comparison_result.statistical_tests)
        
        summary = (
            f"Statistical validation of {num_models} neuromorphic models across {num_metrics} metrics "
            f"shows that '{best_model}' achieves the best overall performance. "
            f"Out of {total_tests} statistical tests conducted, {significant_tests} showed significant differences. "
            f"The analysis confidence level is {confidence:.2%}, based on {comparison_result.sample_size} samples per model."
        )
        
        return summary
    
    def _format_statistical_results(self, statistical_tests: List[StatisticalTest]) -> List[Dict]:
        """Format statistical test results for reporting.
        
        Args:
            statistical_tests: List of statistical test results
            
        Returns:
            Formatted results list
        """
        formatted_results = []
        
        for test in statistical_tests:
            result_dict = {
                "test": test.test_name,
                "statistic": round(test.statistic, 4),
                "p_value": round(test.p_value, 6),
                "significant": test.significant,
                "effect_size": round(test.effect_size, 4) if test.effect_size else None,
                "interpretation": test.interpretation
            }
            
            if test.confidence_interval:
                result_dict["confidence_interval"] = [
                    round(test.confidence_interval[0], 4),
                    round(test.confidence_interval[1], 4)
                ]
            
            formatted_results.append(result_dict)
        
        return formatted_results
    
    def _generate_recommendations(self, comparison_result: ComparisonResult) -> List[str]:
        """Generate actionable recommendations based on results.
        
        Args:
            comparison_result: Comparison results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Best model recommendation
        recommendations.append(
            f"Primary recommendation: Deploy '{comparison_result.best_model}' for production use "
            f"based on superior performance across evaluation metrics."
        )
        
        # Significant differences
        significant_tests = [test for test in comparison_result.statistical_tests if test.significant]
        if significant_tests:
            recommendations.append(
                f"Statistical analysis confirms significant performance differences exist "
                f"between models, validating the model selection process."
            )
        else:
            recommendations.append(
                "No statistically significant differences detected between models. "
                "Consider factors beyond performance metrics for model selection."
            )
        
        # Effect size recommendations
        large_effects = [
            test for test in significant_tests 
            if test.effect_size and abs(test.effect_size) > 0.8
        ]
        if large_effects:
            recommendations.append(
                f"Large effect sizes detected in {len(large_effects)} comparisons, "
                f"indicating practically significant differences beyond statistical significance."
            )
        
        # Sample size recommendations
        if comparison_result.sample_size < 30:
            recommendations.append(
                "Sample size is relatively small. Consider collecting additional data "
                "to increase statistical power and generalizability of results."
            )
        
        return recommendations
    
    def _summarize_convergence(self, learning_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize learning convergence analysis.
        
        Args:
            learning_analysis: Learning analysis results
            
        Returns:
            Convergence summary
        """
        converged_models = [
            model for model, analysis in learning_analysis.items() 
            if analysis["converged"]
        ]
        
        avg_convergence_epoch = np.mean([
            analysis["convergence_epoch"] 
            for analysis in learning_analysis.values() 
            if analysis["converged"]
        ]) if converged_models else 0
        
        most_stable = max(
            learning_analysis.keys(),
            key=lambda x: learning_analysis[x]["stability_score"]
        )
        
        summary = {
            "total_models": len(learning_analysis),
            "converged_models": len(converged_models),
            "convergence_rate": len(converged_models) / len(learning_analysis),
            "average_convergence_epoch": avg_convergence_epoch,
            "most_stable_model": most_stable,
            "converged_model_names": converged_models
        }
        
        return summary
    
    def _generate_validation_plots(
        self,
        comparison_result: ComparisonResult,
        learning_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate validation plots (returns plot data/configs).
        
        Args:
            comparison_result: Comparison results
            learning_analysis: Learning analysis results
            
        Returns:
            Plot configuration dictionary
        """
        plots = {}
        
        # Model performance comparison plot config
        plots["model_comparison"] = {
            "type": "box_plot",
            "title": "Model Performance Comparison",
            "data": comparison_result.rankings,
            "models": comparison_result.models_compared,
            "metrics": comparison_result.metrics_compared
        }
        
        # Effect sizes plot config
        if comparison_result.effect_sizes:
            plots["effect_sizes"] = {
                "type": "bar_plot",
                "title": "Effect Sizes Between Models",
                "data": comparison_result.effect_sizes,
                "threshold_lines": [0.2, 0.5, 0.8]  # Small, medium, large effect
            }
        
        # Statistical significance plot config
        significant_tests = [test for test in comparison_result.statistical_tests if test.significant]
        plots["significance"] = {
            "type": "significance_plot",
            "title": "Statistical Significance Results",
            "total_tests": len(comparison_result.statistical_tests),
            "significant_tests": len(significant_tests),
            "alpha": self.alpha
        }
        
        # Learning curves plot config
        if learning_analysis:
            plots["learning_convergence"] = {
                "type": "line_plot",
                "title": "Learning Convergence Analysis",
                "convergence_data": {
                    model: analysis["convergence_epoch"] 
                    for model, analysis in learning_analysis.items()
                },
                "stability_data": {
                    model: analysis["stability_score"] 
                    for model, analysis in learning_analysis.items()
                }
            }
        
        return plots