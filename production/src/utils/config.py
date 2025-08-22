"""Configuration management for neuromorphic edge processor."""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict, field
from schema import Schema, And, Or, Use, SchemaError


@dataclass
class NeuronConfig:
    """Configuration for neuron parameters."""
    tau_mem: float = 20.0  # ms
    tau_syn: float = 5.0   # ms
    v_thresh: float = -50.0  # mV
    v_reset: float = -70.0   # mV
    v_rest: float = -65.0    # mV
    refractory_period: float = 2.0  # ms
    adaptive_thresh: bool = True
    dt: float = 1.0  # ms


@dataclass
class NetworkConfig:
    """Configuration for network architecture."""
    input_size: int = 784
    hidden_sizes: list = field(default_factory=lambda: [256, 128])
    output_size: int = 10
    learning_rule: str = "stdp"
    topology: str = "feedforward"
    connection_sparsity: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    checkpoint_interval: int = 10
    use_mixed_precision: bool = False


@dataclass
class DeviceConfig:
    """Configuration for device and deployment."""
    device: str = "cpu"
    num_workers: int = 0
    pin_memory: bool = False
    enable_profiling: bool = False
    target_platform: str = "generic"  # generic, raspberry_pi, jetson, loihi
    power_budget_mw: float = 1000.0
    latency_target_ms: float = 10.0


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "neuromorphic_processor.log"
    log_to_console: bool = True
    enable_wandb: bool = False
    wandb_project: str = "neuromorphic-edge"
    enable_tensorboard: bool = True
    metrics_interval: int = 100


@dataclass
class SecurityConfig:
    """Configuration for security measures."""
    enable_input_validation: bool = True
    max_input_size_mb: float = 100.0
    rate_limit_requests_per_minute: int = 1000
    enable_model_encryption: bool = False
    allow_external_data: bool = False
    sanitize_outputs: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "neuromorphic_experiment"
    version: str = "1.0.0"
    description: str = "Neuromorphic edge processing experiment"
    
    # Sub-configurations
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Additional experiment settings
    random_seed: int = 42
    reproducible: bool = True
    save_checkpoints: bool = True
    output_directory: str = "experiments"


class ConfigValidator:
    """Validates configuration parameters with comprehensive schemas."""
    
    def __init__(self):
        self.schemas = self._create_validation_schemas()
    
    def _create_validation_schemas(self) -> Dict[str, Schema]:
        """Create validation schemas for all configuration classes."""
        
        neuron_schema = Schema({
            'tau_mem': And(float, lambda x: x > 0, error='tau_mem must be positive'),
            'tau_syn': And(float, lambda x: x > 0, error='tau_syn must be positive'),
            'v_thresh': float,
            'v_reset': float,
            'v_rest': float,
            'refractory_period': And(float, lambda x: x >= 0, error='refractory_period must be non-negative'),
            'adaptive_thresh': bool,
            'dt': And(float, lambda x: 0 < x <= 10, error='dt must be between 0 and 10 ms')
        })
        
        network_schema = Schema({
            'input_size': And(int, lambda x: x > 0, error='input_size must be positive'),
            'hidden_sizes': And(list, lambda x: all(isinstance(i, int) and i > 0 for i in x)),
            'output_size': And(int, lambda x: x > 0, error='output_size must be positive'),
            'learning_rule': And(str, lambda x: x in ['stdp', 'backprop', 'rl']),
            'topology': And(str, lambda x: x in ['feedforward', 'recurrent', 'random', 'small_world']),
            'connection_sparsity': And(float, lambda x: 0 <= x <= 1, error='sparsity must be [0,1]')
        })
        
        training_schema = Schema({
            'batch_size': And(int, lambda x: x > 0, error='batch_size must be positive'),
            'learning_rate': And(float, lambda x: x > 0, error='learning_rate must be positive'),
            'num_epochs': And(int, lambda x: x > 0, error='num_epochs must be positive'),
            'validation_split': And(float, lambda x: 0 <= x < 1, error='validation_split must be [0,1)'),
            'early_stopping_patience': And(int, lambda x: x >= 0),
            'checkpoint_interval': And(int, lambda x: x > 0),
            'use_mixed_precision': bool
        })
        
        device_schema = Schema({
            'device': And(str, lambda x: x in ['cpu', 'cuda', 'mps']),
            'num_workers': And(int, lambda x: x >= 0),
            'pin_memory': bool,
            'enable_profiling': bool,
            'target_platform': And(str, lambda x: x in ['generic', 'raspberry_pi', 'jetson', 'loihi']),
            'power_budget_mw': And(float, lambda x: x > 0),
            'latency_target_ms': And(float, lambda x: x > 0)
        })
        
        logging_schema = Schema({
            'level': And(str, lambda x: x in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
            'log_to_file': bool,
            'log_file': str,
            'log_to_console': bool,
            'enable_wandb': bool,
            'wandb_project': str,
            'enable_tensorboard': bool,
            'metrics_interval': And(int, lambda x: x > 0)
        })
        
        security_schema = Schema({
            'enable_input_validation': bool,
            'max_input_size_mb': And(float, lambda x: x > 0),
            'rate_limit_requests_per_minute': And(int, lambda x: x > 0),
            'enable_model_encryption': bool,
            'allow_external_data': bool,
            'sanitize_outputs': bool
        })
        
        return {
            'neuron': neuron_schema,
            'network': network_schema,
            'training': training_schema,
            'device': device_schema,
            'logging': logging_schema,
            'security': security_schema
        }
    
    def validate_config(self, config_dict: Dict[str, Any], config_type: str) -> bool:
        """Validate configuration dictionary against schema.
        
        Args:
            config_dict: Configuration to validate
            config_type: Type of configuration ('neuron', 'network', etc.)
            
        Returns:
            True if valid, raises SchemaError if invalid
        """
        if config_type not in self.schemas:
            raise ValueError(f"Unknown configuration type: {config_type}")
        
        try:
            self.schemas[config_type].validate(config_dict)
            return True
        except SchemaError as e:
            logging.error(f"Configuration validation failed for {config_type}: {e}")
            raise
    
    def validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """Validate complete experiment configuration."""
        try:
            # Validate each sub-configuration
            self.validate_config(asdict(config.neuron), 'neuron')
            self.validate_config(asdict(config.network), 'network') 
            self.validate_config(asdict(config.training), 'training')
            self.validate_config(asdict(config.device), 'device')
            self.validate_config(asdict(config.logging), 'logging')
            self.validate_config(asdict(config.security), 'security')
            
            # Cross-validation checks
            self._cross_validate(config)
            
            return True
        except (SchemaError, ValueError) as e:
            logging.error(f"Experiment configuration validation failed: {e}")
            raise
    
    def _cross_validate(self, config: ExperimentConfig):
        """Perform cross-validation checks between configuration sections."""
        
        # Check neuron time step vs training settings
        if config.neuron.dt > 5.0 and config.training.num_epochs > 1000:
            logging.warning("Large time step with many epochs may lead to instability")
        
        # Check device compatibility
        if config.device.device == 'cuda' and config.device.target_platform == 'raspberry_pi':
            raise ValueError("CUDA device incompatible with Raspberry Pi platform")
        
        # Check memory requirements
        total_neurons = config.network.input_size + sum(config.network.hidden_sizes) + config.network.output_size
        estimated_memory_mb = total_neurons * 0.001  # Rough estimate
        
        if config.device.target_platform == 'raspberry_pi' and estimated_memory_mb > 500:
            logging.warning(f"Large network ({total_neurons} neurons) may exceed Raspberry Pi memory limits")
        
        # Check security settings
        if config.security.allow_external_data and not config.security.enable_input_validation:
            logging.warning("External data allowed without input validation - security risk")


class ConfigManager:
    """Advanced configuration management with validation, templating, and environment support."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files (default: ./configs)
        """
        self.config_dir = Path(config_dir or "./configs")
        self.config_dir.mkdir(exist_ok=True)
        
        self.validator = ConfigValidator()
        self.logger = logging.getLogger(__name__)
        
        # Environment variable prefix
        self.env_prefix = "NEUROMORPHIC_"
        
    def create_default_config(self) -> ExperimentConfig:
        """Create default experiment configuration."""
        return ExperimentConfig()
    
    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from file with comprehensive error handling.
        
        Args:
            config_path: Path to configuration file (.json or .yaml)
            
        Returns:
            Loaded and validated ExperimentConfig
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
            SchemaError: If config validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            # Load based on file extension
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            # Apply environment variable overrides
            config_dict = self._apply_env_overrides(config_dict)
            
            # Convert to ExperimentConfig
            config = self._dict_to_config(config_dict)
            
            # Validate configuration
            self.validator.validate_experiment_config(config)
            
            self.logger.info(f"Successfully loaded configuration from {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {config_path}: {e}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def save_config(self, config: ExperimentConfig, config_path: Union[str, Path], format: str = "yaml"):
        """Save configuration to file with backup.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
            format: File format ('json' or 'yaml')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists
        if config_path.exists():
            backup_path = config_path.with_suffix(f".backup{config_path.suffix}")
            config_path.rename(backup_path)
            self.logger.info(f"Created backup: {backup_path}")
        
        try:
            config_dict = asdict(config)
            
            if format.lower() == 'json':
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            elif format.lower() in ['yaml', 'yml']:
                with open(config_path, 'w') as f:
                    yaml.safe_dump(config_dict, f, indent=2, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        
        def apply_override(obj, key_path, value):
            """Recursively apply override to nested dictionary."""
            keys = key_path.split('.')
            current = obj
            
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set final value with type conversion
            final_key = keys[-1]
            if final_key in current:
                # Attempt to preserve original type
                original_value = current[final_key]
                if isinstance(original_value, bool):
                    current[final_key] = value.lower() in ['true', '1', 'yes']
                elif isinstance(original_value, int):
                    current[final_key] = int(value)
                elif isinstance(original_value, float):
                    current[final_key] = float(value)
                else:
                    current[final_key] = value
            else:
                current[final_key] = value
        
        # Check for environment variable overrides
        for env_var, value in os.environ.items():
            if env_var.startswith(self.env_prefix):
                # Convert NEUROMORPHIC_NETWORK_HIDDEN_SIZES -> network.hidden_sizes
                config_key = env_var[len(self.env_prefix):].lower().replace('_', '.')
                
                try:
                    apply_override(config_dict, config_key, value)
                    self.logger.info(f"Applied environment override: {config_key} = {value}")
                except Exception as e:
                    self.logger.warning(f"Failed to apply environment override {env_var}: {e}")
        
        return config_dict
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig object."""
        
        # Extract sub-configurations
        neuron_config = NeuronConfig(**config_dict.get('neuron', {}))
        network_config = NetworkConfig(**config_dict.get('network', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        device_config = DeviceConfig(**config_dict.get('device', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        security_config = SecurityConfig(**config_dict.get('security', {}))
        
        # Create main config
        main_config_dict = {k: v for k, v in config_dict.items() 
                           if k not in ['neuron', 'network', 'training', 'device', 'logging', 'security']}
        
        return ExperimentConfig(
            **main_config_dict,
            neuron=neuron_config,
            network=network_config,
            training=training_config,
            device=device_config,
            logging=logging_config,
            security=security_config
        )
    
    def create_config_templates(self):
        """Create configuration templates for different use cases."""
        
        templates = {
            "basic_classification": ExperimentConfig(
                name="basic_classification",
                description="Basic neuromorphic classification",
                network=NetworkConfig(
                    input_size=784,
                    hidden_sizes=[128, 64],
                    output_size=10
                ),
                training=TrainingConfig(
                    batch_size=32,
                    num_epochs=50
                )
            ),
            
            "edge_deployment": ExperimentConfig(
                name="edge_deployment",
                description="Edge device deployment configuration",
                network=NetworkConfig(
                    input_size=784,
                    hidden_sizes=[64, 32],
                    output_size=10,
                    connection_sparsity=0.3
                ),
                device=DeviceConfig(
                    target_platform="raspberry_pi",
                    power_budget_mw=500.0,
                    latency_target_ms=5.0
                ),
                security=SecurityConfig(
                    enable_input_validation=True,
                    max_input_size_mb=10.0
                )
            ),
            
            "research_experiment": ExperimentConfig(
                name="research_experiment",
                description="Research configuration with detailed logging",
                training=TrainingConfig(
                    batch_size=16,
                    num_epochs=200,
                    use_mixed_precision=True
                ),
                logging=LoggingConfig(
                    level="DEBUG",
                    enable_wandb=True,
                    enable_tensorboard=True
                ),
                device=DeviceConfig(
                    enable_profiling=True
                )
            )
        }
        
        # Save templates
        for name, config in templates.items():
            template_path = self.config_dir / "templates" / f"{name}.yaml"
            template_path.parent.mkdir(exist_ok=True)
            self.save_config(config, template_path, format="yaml")
        
        self.logger.info(f"Created {len(templates)} configuration templates in {self.config_dir / 'templates'}")
    
    def get_platform_optimized_config(self, platform: str) -> ExperimentConfig:
        """Get platform-optimized configuration.
        
        Args:
            platform: Target platform ('raspberry_pi', 'jetson', 'loihi', etc.)
            
        Returns:
            Optimized configuration for platform
        """
        base_config = self.create_default_config()
        
        if platform == "raspberry_pi":
            base_config.network.hidden_sizes = [64, 32]  # Smaller network
            base_config.training.batch_size = 8  # Smaller batches
            base_config.device.target_platform = "raspberry_pi"
            base_config.device.power_budget_mw = 500.0
            base_config.device.num_workers = 2
            
        elif platform == "jetson":
            base_config.network.hidden_sizes = [128, 64]
            base_config.training.batch_size = 16
            base_config.device.target_platform = "jetson"
            base_config.device.device = "cuda"
            base_config.device.power_budget_mw = 2000.0
            
        elif platform == "loihi":
            base_config.network.topology = "random"
            base_config.network.connection_sparsity = 0.2
            base_config.neuron.adaptive_thresh = True
            base_config.device.target_platform = "loihi"
            base_config.device.power_budget_mw = 100.0  # Very low power
            
        return base_config