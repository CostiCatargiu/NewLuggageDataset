# Ultralytics Attention Debug Utilities
"""
Debug utilities for tracking attention module behavior during training.

Usage:
    from ultralytics.utils.debug_attention import AttentionDebugger
    
    # Initialize (call once at training start)
    debugger = AttentionDebugger(log_every=100)  # Log every 100 batches
    
    # In your attention module forward():
    debugger.log_attention_stats(name="P3_CBAM", channel_weights=w, spatial_weights=s)
    debugger.log_scale_selection(name="P3_spatial", scale_weights=scale_weights)
    
    # At end of epoch:
    debugger.summarize_epoch(epoch=1)
    debugger.reset()
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, List
import json
import os


class AttentionDebugger:
    """
    Tracks and logs attention module statistics during training.
    
    Key metrics tracked:
    1. Channel attention: Which channels are amplified/suppressed
    2. Spatial attention: Where the model focuses
    3. Scale selection: Which kernel size (3x3/7x7/11x11) is preferred
    4. Residual weight: How much original vs attention output is used
    5. Per-class performance indicators
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_every: int = 100, save_dir: str = None, enabled: bool = True):
        if self._initialized:
            return
            
        self.log_every = log_every
        self.save_dir = save_dir or "debug_logs"
        self.enabled = enabled
        self.batch_count = 0
        self.epoch = 0
        
        # Storage for statistics
        self.channel_stats = defaultdict(list)      # {name: [mean_weights]}
        self.spatial_stats = defaultdict(list)      # {name: [mean_activation]}
        self.scale_stats = defaultdict(list)        # {name: [[w3, w7, w11], ...]}
        self.residual_stats = defaultdict(list)     # {name: [alpha_values]}
        self.gradient_stats = defaultdict(list)     # {name: [grad_norm]}
        
        # Per-size tracking (if available)
        self.size_losses = defaultdict(list)        # {small/medium/large: [losses]}
        self.class_losses = defaultdict(list)       # {class_name: [losses]}
        
        self._initialized = True
        
        if enabled:
            os.makedirs(save_dir, exist_ok=True)
            print(f"[AttentionDebugger] Initialized. Logging every {log_every} batches to {save_dir}/")
    
    def log_channel_attention(self, name: str, weights: torch.Tensor):
        """
        Log channel attention weights.
        
        Args:
            name: Module name (e.g., "P3_CBAM")
            weights: Channel weights [B, C, 1, 1] after sigmoid
        """
        if not self.enabled:
            return
            
        with torch.no_grad():
            # Statistics about channel weights
            mean_w = weights.mean().item()
            std_w = weights.std().item()
            max_w = weights.max().item()
            min_w = weights.min().item()
            
            # How many channels are "active" (> 0.5) vs "suppressed" (< 0.5)
            active_ratio = (weights > 0.5).float().mean().item()
            
            self.channel_stats[name].append({
                'mean': mean_w,
                'std': std_w,
                'max': max_w,
                'min': min_w,
                'active_ratio': active_ratio
            })
            
            if self.batch_count % self.log_every == 0:
                print(f"[Debug] {name} Channel: mean={mean_w:.3f}, std={std_w:.3f}, "
                      f"active={active_ratio:.1%}")
    
    def log_spatial_attention(self, name: str, weights: torch.Tensor):
        """
        Log spatial attention weights.
        
        Args:
            name: Module name
            weights: Spatial weights [B, 1, H, W] after sigmoid
        """
        if not self.enabled:
            return
            
        with torch.no_grad():
            mean_w = weights.mean().item()
            std_w = weights.std().item()
            
            # Spatial focus: how concentrated is attention?
            # High std = focused on specific regions; Low std = uniform
            focus_score = std_w / (mean_w + 1e-6)
            
            self.spatial_stats[name].append({
                'mean': mean_w,
                'std': std_w,
                'focus_score': focus_score
            })
            
            if self.batch_count % self.log_every == 0:
                print(f"[Debug] {name} Spatial: mean={mean_w:.3f}, focus={focus_score:.3f}")
    
    def log_scale_selection(self, name: str, scale_weights: torch.Tensor):
        """
        Log scale selection in ScaleAdaptiveSpatialAttention.
        
        Args:
            name: Module name
            scale_weights: [B, 3, 1, 1] weights for 3x3, 7x7, 11x11
        """
        if not self.enabled:
            return
            
        with torch.no_grad():
            # Average across batch
            w = scale_weights.mean(dim=0).squeeze()  # [3]
            w3, w7, w11 = w[0].item(), w[1].item(), w[2].item()
            
            # Which scale dominates?
            dominant = ['3x3 (fine)', '7x7 (medium)', '11x11 (large)'][w.argmax().item()]
            
            self.scale_stats[name].append({
                'w3': w3, 'w7': w7, 'w11': w11,
                'dominant': dominant
            })
            
            if self.batch_count % self.log_every == 0:
                print(f"[Debug] {name} Scale: 3x3={w3:.2f}, 7x7={w7:.2f}, 11x11={w11:.2f} → {dominant}")
    
    def log_residual_weight(self, name: str, alpha: torch.Tensor):
        """
        Log learnable residual weight.
        
        Args:
            name: Module name
            alpha: Residual weight after sigmoid [scalar or tensor]
        """
        if not self.enabled:
            return
            
        with torch.no_grad():
            alpha_val = alpha.mean().item() if alpha.numel() > 1 else alpha.item()
            
            self.residual_stats[name].append(alpha_val)
            
            if self.batch_count % self.log_every == 0:
                # alpha > 0.5 means attention dominates; < 0.5 means identity dominates
                mode = "attention" if alpha_val > 0.5 else "identity"
                print(f"[Debug] {name} Residual α={alpha_val:.3f} ({mode} dominant)")
    
    def log_loss_by_size(self, size: str, loss: float):
        """
        Log loss for different object sizes.
        
        Args:
            size: 'small', 'medium', or 'large'
            loss: Loss value for this size category
        """
        if not self.enabled:
            return
        self.size_losses[size].append(loss)
    
    def log_loss_by_class(self, class_name: str, loss: float):
        """
        Log loss for different classes.
        
        Args:
            class_name: 'backpack', 'bag', or 'trolley'
            loss: Loss value for this class
        """
        if not self.enabled:
            return
        self.class_losses[class_name].append(loss)
    
    def log_gradient_norm(self, name: str, module: torch.nn.Module):
        """
        Log gradient norm for a module (call after backward).
        
        Args:
            name: Module name
            module: The nn.Module to check gradients for
        """
        if not self.enabled:
            return
            
        total_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        self.gradient_stats[name].append(total_norm)
        
        if self.batch_count % self.log_every == 0:
            print(f"[Debug] {name} Gradient norm: {total_norm:.4f}")
    
    def step(self):
        """Call after each batch to increment counter."""
        self.batch_count += 1
    
    def summarize_epoch(self, epoch: int):
        """
        Print and save epoch summary.
        
        Args:
            epoch: Current epoch number
        """
        if not self.enabled:
            return
            
        self.epoch = epoch
        print(f"\n{'='*60}")
        print(f"[AttentionDebugger] Epoch {epoch} Summary")
        print(f"{'='*60}")
        
        summary = {'epoch': epoch, 'batches': self.batch_count}
        
        # Channel attention summary
        if self.channel_stats:
            print("\n📊 Channel Attention:")
            summary['channel'] = {}
            for name, stats in self.channel_stats.items():
                if stats:
                    avg_mean = np.mean([s['mean'] for s in stats])
                    avg_active = np.mean([s['active_ratio'] for s in stats])
                    print(f"  {name}: avg_weight={avg_mean:.3f}, active_channels={avg_active:.1%}")
                    summary['channel'][name] = {'mean': avg_mean, 'active_ratio': avg_active}
        
        # Scale selection summary
        if self.scale_stats:
            print("\n📐 Scale Selection:")
            summary['scale'] = {}
            for name, stats in self.scale_stats.items():
                if stats:
                    avg_w3 = np.mean([s['w3'] for s in stats])
                    avg_w7 = np.mean([s['w7'] for s in stats])
                    avg_w11 = np.mean([s['w11'] for s in stats])
                    dominant_counts = defaultdict(int)
                    for s in stats:
                        dominant_counts[s['dominant']] += 1
                    most_common = max(dominant_counts, key=dominant_counts.get)
                    print(f"  {name}: 3x3={avg_w3:.2f}, 7x7={avg_w7:.2f}, 11x11={avg_w11:.2f}")
                    print(f"         Most common: {most_common} ({dominant_counts[most_common]}/{len(stats)} batches)")
                    summary['scale'][name] = {'w3': avg_w3, 'w7': avg_w7, 'w11': avg_w11, 'dominant': most_common}
        
        # Residual weight summary
        if self.residual_stats:
            print("\n⚖️ Residual Weights:")
            summary['residual'] = {}
            for name, alphas in self.residual_stats.items():
                if alphas:
                    avg_alpha = np.mean(alphas)
                    trend = "↑" if len(alphas) > 1 and alphas[-1] > alphas[0] else "↓" if len(alphas) > 1 and alphas[-1] < alphas[0] else "→"
                    print(f"  {name}: α={avg_alpha:.3f} {trend} ({'attention' if avg_alpha > 0.5 else 'identity'} dominant)")
                    summary['residual'][name] = avg_alpha
        
        # Loss by size
        if self.size_losses:
            print("\n📏 Loss by Size:")
            summary['size_loss'] = {}
            for size, losses in self.size_losses.items():
                if losses:
                    avg_loss = np.mean(losses)
                    print(f"  {size}: {avg_loss:.4f}")
                    summary['size_loss'][size] = avg_loss
        
        # Loss by class
        if self.class_losses:
            print("\n🏷️ Loss by Class:")
            summary['class_loss'] = {}
            for cls, losses in self.class_losses.items():
                if losses:
                    avg_loss = np.mean(losses)
                    print(f"  {cls}: {avg_loss:.4f}")
                    summary['class_loss'][cls] = avg_loss
        
        # Gradient stats
        if self.gradient_stats:
            print("\n📈 Gradient Norms:")
            summary['gradients'] = {}
            for name, norms in self.gradient_stats.items():
                if norms:
                    avg_norm = np.mean(norms)
                    max_norm = np.max(norms)
                    print(f"  {name}: avg={avg_norm:.4f}, max={max_norm:.4f}")
                    summary['gradients'][name] = {'avg': avg_norm, 'max': max_norm}
        
        print(f"{'='*60}\n")
        
        # Save to file
        save_path = os.path.join(self.save_dir, f"epoch_{epoch}_debug.json")
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[AttentionDebugger] Saved to {save_path}")
        
        return summary
    
    def reset(self):
        """Reset all statistics for new epoch."""
        self.channel_stats.clear()
        self.spatial_stats.clear()
        self.scale_stats.clear()
        self.residual_stats.clear()
        self.gradient_stats.clear()
        self.size_losses.clear()
        self.class_losses.clear()
        self.batch_count = 0
    
    def disable(self):
        """Disable debugging."""
        self.enabled = False
    
    def enable(self):
        """Enable debugging."""
        self.enabled = True


# Global instance for easy access
_debugger: Optional[AttentionDebugger] = None


def get_debugger(log_every: int = 100, save_dir: str = "debug_logs", enabled: bool = True) -> AttentionDebugger:
    """Get or create the global debugger instance."""
    global _debugger
    if _debugger is None:
        _debugger = AttentionDebugger(log_every=log_every, save_dir=save_dir, enabled=enabled)
    return _debugger


def log_channel(name: str, weights: torch.Tensor):
    """Shortcut for logging channel attention."""
    if _debugger:
        _debugger.log_channel_attention(name, weights)


def log_spatial(name: str, weights: torch.Tensor):
    """Shortcut for logging spatial attention."""
    if _debugger:
        _debugger.log_spatial_attention(name, weights)


def log_scale(name: str, weights: torch.Tensor):
    """Shortcut for logging scale selection."""
    if _debugger:
        _debugger.log_scale_selection(name, weights)


def log_residual(name: str, alpha: torch.Tensor):
    """Shortcut for logging residual weight."""
    if _debugger:
        _debugger.log_residual_weight(name, alpha)
