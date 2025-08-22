"""Visualization utilities for neuromorphic computing and spiking neural networks."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import networkx as nx


class Visualizer:
    """General-purpose visualizer for neuromorphic data and results."""
    
    def __init__(self, style: str = "neuromorphic", figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer.
        
        Args:
            style: Plotting style ('neuromorphic', 'scientific', 'minimal')
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style based on chosen theme."""
        if self.style == "neuromorphic":
            plt.style.use('dark_background')
            self.colors = {
                'spike': '#00ff00',
                'membrane': '#ffaa00', 
                'threshold': '#ff0000',
                'current': '#00aaff',
                'background': '#000000',
                'grid': '#333333'
            }
        elif self.style == "scientific":
            plt.style.use('seaborn-v0_8')
            self.colors = {
                'spike': '#2e8b57',
                'membrane': '#ff6347',
                'threshold': '#dc143c', 
                'current': '#4682b4',
                'background': '#ffffff',
                'grid': '#cccccc'
            }
        else:  # minimal
            plt.style.use('default')
            self.colors = {
                'spike': '#000000',
                'membrane': '#666666',
                'threshold': '#999999',
                'current': '#333333',
                'background': '#ffffff',
                'grid': '#eeeeee'
            }
    
    def plot_raster(
        self,
        spike_trains: torch.Tensor,
        dt: float = 1.0,
        neuron_ids: Optional[List[int]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        title: str = "Spike Raster Plot",
        save_path: Optional[str] = None
    ):
        """Create a raster plot of spike trains.
        
        Args:
            spike_trains: Spike data [num_neurons, time_steps] or [batch, num_neurons, time_steps]
            dt: Time step in ms
            neuron_ids: Specific neuron IDs to plot
            time_range: Time range to display (start_ms, end_ms)
            title: Plot title
            save_path: Path to save the plot
        """
        # Handle batch dimension
        if spike_trains.dim() == 3:
            spike_trains = spike_trains[0]  # Take first batch
        
        num_neurons, time_steps = spike_trains.shape
        time_axis = torch.arange(time_steps) * dt
        
        # Apply time range filter
        if time_range is not None:
            start_idx = int(time_range[0] / dt)
            end_idx = int(time_range[1] / dt)
            time_axis = time_axis[start_idx:end_idx]
            spike_trains = spike_trains[:, start_idx:end_idx]
        
        # Select neurons to plot
        if neuron_ids is None:
            neuron_ids = range(min(num_neurons, 100))  # Limit to 100 neurons
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot spikes for each neuron
        for i, neuron_id in enumerate(neuron_ids):
            if neuron_id < num_neurons:
                spike_times = time_axis[spike_trains[neuron_id] > 0]
                if len(spike_times) > 0:
                    ax.scatter(spike_times, [neuron_id] * len(spike_times), 
                             s=2, c=self.colors['spike'], alpha=0.8, marker='|')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron ID')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_membrane_potential(
        self,
        membrane_data: torch.Tensor,
        threshold: float = 1.0,
        dt: float = 1.0,
        neuron_ids: Optional[List[int]] = None,
        spike_times: Optional[torch.Tensor] = None,
        title: str = "Membrane Potential",
        save_path: Optional[str] = None
    ):
        """Plot membrane potential traces with optional spike markers.
        
        Args:
            membrane_data: Membrane potential [num_neurons, time_steps]
            threshold: Spike threshold value
            dt: Time step in ms
            neuron_ids: Specific neurons to plot
            spike_times: Spike times for marking [num_neurons, time_steps]
            title: Plot title
            save_path: Path to save the plot
        """
        num_neurons, time_steps = membrane_data.shape
        time_axis = torch.arange(time_steps) * dt
        
        if neuron_ids is None:
            neuron_ids = range(min(num_neurons, 5))  # Plot up to 5 neurons
        
        fig, axes = plt.subplots(len(neuron_ids), 1, figsize=(self.figsize[0], 2*len(neuron_ids)), 
                                sharex=True)
        if len(neuron_ids) == 1:
            axes = [axes]
        
        for i, neuron_id in enumerate(neuron_ids):
            if neuron_id < num_neurons:
                # Plot membrane potential
                axes[i].plot(time_axis, membrane_data[neuron_id], 
                           color=self.colors['membrane'], linewidth=1.5, 
                           label=f'Neuron {neuron_id}')
                
                # Plot threshold line
                axes[i].axhline(y=threshold, color=self.colors['threshold'], 
                              linestyle='--', alpha=0.7, label='Threshold')
                
                # Mark spike times if provided
                if spike_times is not None:
                    spikes = time_axis[spike_times[neuron_id] > 0]
                    if len(spikes) > 0:
                        spike_potentials = [threshold + 0.1] * len(spikes)
                        axes[i].scatter(spikes, spike_potentials, 
                                      color=self.colors['spike'], s=30, 
                                      marker='v', label='Spikes')
                
                axes[i].set_ylabel('Membrane\nPotential (mV)')
                axes[i].legend(loc='upper right')
                axes[i].grid(True, alpha=0.3, color=self.colors['grid'])
        
        axes[-1].set_xlabel('Time (ms)')
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_firing_rates(
        self,
        spike_trains: torch.Tensor,
        window_size: float = 50.0,
        dt: float = 1.0,
        neuron_ids: Optional[List[int]] = None,
        title: str = "Instantaneous Firing Rates",
        save_path: Optional[str] = None
    ):
        """Plot instantaneous firing rates using sliding window.
        
        Args:
            spike_trains: Spike data [num_neurons, time_steps]
            window_size: Sliding window size in ms
            dt: Time step in ms
            neuron_ids: Specific neurons to plot
            title: Plot title
            save_path: Path to save the plot
        """
        if spike_trains.dim() == 3:
            spike_trains = spike_trains[0]
        
        num_neurons, time_steps = spike_trains.shape
        window_steps = int(window_size / dt)
        
        # Compute firing rates using convolution
        kernel = torch.ones(window_steps) / window_size * 1000.0  # Convert to Hz
        rates = torch.zeros_like(spike_trains, dtype=torch.float32)
        
        for i in range(num_neurons):
            rates[i] = torch.conv1d(
                spike_trains[i].unsqueeze(0).unsqueeze(0).float(),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=window_steps//2
            ).squeeze()
        
        time_axis = torch.arange(time_steps) * dt
        
        if neuron_ids is None:
            neuron_ids = range(min(num_neurons, 10))
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(neuron_ids)))
        
        for i, neuron_id in enumerate(neuron_ids):
            if neuron_id < num_neurons:
                ax.plot(time_axis, rates[neuron_id], 
                       color=colors[i], linewidth=1.5, 
                       label=f'Neuron {neuron_id}', alpha=0.8)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_weight_matrix(
        self,
        weights: torch.Tensor,
        title: str = "Synaptic Weight Matrix",
        cmap: str = "viridis",
        save_path: Optional[str] = None
    ):
        """Visualize synaptic weight matrix as heatmap.
        
        Args:
            weights: Weight matrix [output_neurons, input_neurons]
            title: Plot title
            cmap: Colormap name
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(weights.detach().cpu().numpy(), cmap=cmap, aspect='auto')
        
        ax.set_xlabel('Input Neurons')
        ax.set_ylabel('Output Neurons')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Weight Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spike_statistics(
        self,
        spike_trains: torch.Tensor,
        dt: float = 1.0,
        bins: int = 50,
        save_path: Optional[str] = None
    ):
        """Plot various spike train statistics.
        
        Args:
            spike_trains: Spike data [num_neurons, time_steps] 
            dt: Time step in ms
            bins: Number of histogram bins
            save_path: Path to save the plot
        """
        if spike_trains.dim() == 3:
            spike_trains = spike_trains[0]
        
        num_neurons, time_steps = spike_trains.shape
        duration = time_steps * dt / 1000.0  # Duration in seconds
        
        # Calculate statistics
        spike_counts = spike_trains.sum(dim=1)
        firing_rates = spike_counts / duration
        
        # Inter-spike intervals
        all_isis = []
        for neuron in range(num_neurons):
            spike_times = torch.where(spike_trains[neuron] > 0)[0] * dt
            if len(spike_times) > 1:
                isis = torch.diff(spike_times)
                all_isis.extend(isis.tolist())
        
        # Network synchrony (coefficient of variation of population activity)
        pop_activity = spike_trains.sum(dim=0)  # Sum across neurons
        cv_sync = pop_activity.std() / (pop_activity.mean() + 1e-8)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Firing rate distribution
        axes[0, 0].hist(firing_rates.cpu().numpy(), bins=bins, alpha=0.7, 
                       color=self.colors['spike'], edgecolor='black')
        axes[0, 0].set_xlabel('Firing Rate (Hz)')
        axes[0, 0].set_ylabel('Number of Neurons')
        axes[0, 0].set_title('Firing Rate Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ISI distribution
        if all_isis:
            axes[0, 1].hist(all_isis, bins=bins, alpha=0.7, 
                           color=self.colors['membrane'], edgecolor='black')
            axes[0, 1].set_xlabel('Inter-Spike Interval (ms)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('ISI Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Population activity over time
        time_axis = torch.arange(time_steps) * dt
        axes[1, 0].plot(time_axis, pop_activity, color=self.colors['current'], linewidth=1)
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Population Activity')
        axes[1, 0].set_title(f'Network Activity (CV = {cv_sync:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spike count vs neuron ID
        neuron_ids = torch.arange(num_neurons)
        axes[1, 1].bar(neuron_ids[:min(50, num_neurons)], 
                       spike_counts[:min(50, num_neurons)],
                       color=self.colors['threshold'], alpha=0.7)
        axes[1, 1].set_xlabel('Neuron ID')
        axes[1, 1].set_ylabel('Spike Count')
        axes[1, 1].set_title('Spike Counts by Neuron')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class NetworkVisualizer:
    """Specialized visualizer for neural network architectures and dynamics."""
    
    def __init__(self):
        """Initialize network visualizer."""
        pass
    
    def plot_network_architecture(
        self,
        layer_sizes: List[int],
        layer_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Visualize network architecture.
        
        Args:
            layer_sizes: Number of neurons in each layer
            layer_names: Names for each layer
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if layer_names is None:
            layer_names = [f"Layer {i}" for i in range(len(layer_sizes))]
        
        # Calculate positions
        layer_positions = np.linspace(0.1, 0.9, len(layer_sizes))
        max_neurons = max(layer_sizes)
        
        # Draw layers
        for i, (size, name, x_pos) in enumerate(zip(layer_sizes, layer_names, layer_positions)):
            # Calculate neuron positions in this layer
            if size == 1:
                y_positions = [0.5]
            else:
                y_positions = np.linspace(0.1, 0.9, size)
            
            # Draw neurons
            for y_pos in y_positions:
                circle = plt.Circle((x_pos, y_pos), 0.02, 
                                  color='lightblue', ec='black', linewidth=1)
                ax.add_patch(circle)
            
            # Add layer label
            ax.text(x_pos, 0.05, f"{name}\n({size} neurons)", 
                   ha='center', va='bottom', fontsize=10, weight='bold')
            
            # Draw connections to next layer
            if i < len(layer_sizes) - 1:
                next_size = layer_sizes[i + 1]
                next_x = layer_positions[i + 1]
                
                if next_size == 1:
                    next_y_positions = [0.5]
                else:
                    next_y_positions = np.linspace(0.1, 0.9, next_size)
                
                # Draw sample connections (not all to avoid clutter)
                n_connections = min(size * next_size, 50)  # Limit connections
                
                for _ in range(n_connections):
                    start_y = np.random.choice(y_positions)
                    end_y = np.random.choice(next_y_positions)
                    
                    ax.plot([x_pos + 0.02, next_x - 0.02], [start_y, end_y],
                           'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Spiking Neural Network Architecture', fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def animate_network_activity(
        self,
        spike_trains: torch.Tensor,
        layer_sizes: List[int],
        dt: float = 1.0,
        save_path: Optional[str] = None,
        fps: int = 10
    ):
        """Create animation of network spiking activity.
        
        Args:
            spike_trains: Spike data [total_neurons, time_steps]
            layer_sizes: Neurons per layer
            dt: Time step in ms
            save_path: Path to save animation
            fps: Frames per second
        """
        total_neurons, time_steps = spike_trains.shape
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate positions for all neurons
        layer_positions = np.linspace(0.1, 0.9, len(layer_sizes))
        neuron_positions = []
        layer_indices = []
        
        neuron_idx = 0
        for layer_idx, (size, x_pos) in enumerate(zip(layer_sizes, layer_positions)):
            if size == 1:
                y_positions = [0.5]
            else:
                y_positions = np.linspace(0.1, 0.9, size)
            
            for y_pos in y_positions:
                if neuron_idx < total_neurons:
                    neuron_positions.append((x_pos, y_pos))
                    layer_indices.append(layer_idx)
                    neuron_idx += 1
        
        # Initialize scatter plot
        colors = ['lightblue'] * len(neuron_positions)
        sizes = [100] * len(neuron_positions)
        
        scat = ax.scatter([pos[0] for pos in neuron_positions], 
                         [pos[1] for pos in neuron_positions],
                         c=colors, s=sizes, alpha=0.8)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Animation function
        def animate(frame):
            current_spikes = spike_trains[:, frame]
            
            # Update colors based on spiking
            new_colors = []
            new_sizes = []
            
            for i in range(len(neuron_positions)):
                if i < len(current_spikes) and current_spikes[i] > 0:
                    new_colors.append('red')  # Spiking neurons are red
                    new_sizes.append(200)     # Larger size for spiking
                else:
                    new_colors.append('lightblue')
                    new_sizes.append(100)
            
            scat.set_color(new_colors)
            scat.set_sizes(new_sizes)
            
            ax.set_title(f'Network Activity at t = {frame * dt:.1f} ms', 
                        fontsize=14, weight='bold')
            
            return scat,
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=time_steps, interval=1000//fps, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
            plt.close()
        else:
            plt.show()
        
        return anim
    
    def plot_connectivity_graph(
        self,
        weights: torch.Tensor,
        threshold: float = 0.1,
        layout: str = "spring",
        save_path: Optional[str] = None
    ):
        """Plot network connectivity as a graph.
        
        Args:
            weights: Weight matrix [output_neurons, input_neurons]
            threshold: Minimum weight to show connection
            layout: Graph layout algorithm
            save_path: Path to save the plot
        """
        # Create networkx graph
        G = nx.DiGraph()
        
        output_size, input_size = weights.shape
        
        # Add nodes
        for i in range(input_size):
            G.add_node(f"I{i}", layer="input")
        for i in range(output_size):
            G.add_node(f"O{i}", layer="output")
        
        # Add edges above threshold
        weights_np = weights.detach().cpu().numpy()
        for i in range(output_size):
            for j in range(input_size):
                if abs(weights_np[i, j]) > threshold:
                    G.add_edge(f"I{j}", f"O{i}", weight=weights_np[i, j])
        
        # Create layout
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "bipartite":
            input_nodes = [n for n in G.nodes() if n.startswith('I')]
            output_nodes = [n for n in G.nodes() if n.startswith('O')]
            pos = {}
            
            # Position input nodes on left
            for i, node in enumerate(input_nodes):
                pos[node] = (0, i / max(1, len(input_nodes) - 1))
            
            # Position output nodes on right  
            for i, node in enumerate(output_nodes):
                pos[node] = (1, i / max(1, len(output_nodes) - 1))
        else:
            pos = nx.random_layout(G)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw nodes
        input_nodes = [n for n in G.nodes() if n.startswith('I')]
        output_nodes = [n for n in G.nodes() if n.startswith('O')]
        
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, 
                              node_color='lightgreen', node_size=300, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes,
                              node_color='lightcoral', node_size=300, ax=ax)
        
        # Draw edges with weights as thickness
        edges = G.edges()
        edge_weights = [abs(G[u][v]['weight']) for u, v in edges]
        
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [w / max_weight * 3 for w in edge_weights]
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                                  edge_color='gray', arrows=True, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title('Network Connectivity Graph')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()