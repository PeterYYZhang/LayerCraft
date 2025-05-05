"""
LoRA Controller Module

This module provides context managers for controlling LoRA (Low-Rank Adaptation) 
behavior during neural network execution. It enables fine-grained control over which
LoRA adapters are active and their scaling factors during specific operations.
"""

from peft.tuners.tuners_utils import BaseTunerLayer
from typing import List, Any, Optional, Type, Union, Dict
import torch.nn as nn


class enable_lora:
    """
    Context manager that enables or disables specific LoRA adapters in modules.
    
    This class temporarily modifies the active adapters and their scaling factors
    for the duration of the context. When exiting the context, the original state
    of the adapters is restored.
    
    Typical use cases:
    1. Enabling a specific LoRA adapter while disabling others
    2. Temporarily activating all adapters
    3. Controlling which adapters are used during specific operations
    
    Example usage:
    ```python
    with enable_lora(model.modules, adapter_name="adapter1"):
        # Only adapter1 will be active during this block
        output = model(input)
    # Outside the block, original adapter settings are restored
    ```
    """
    
    def __init__(
        self, 
        modules: Union[List[nn.Module], nn.Module], 
        activated: bool = True,
        adapter_name: Optional[str] = None
    ) -> None:
        """
        Initialize the enable_lora context manager.
        
        Args:
            modules: A single module or list of modules to apply LoRA control to.
                     Only modules that are instances of BaseTunerLayer will be affected.
            activated: Whether LoRA functionality should be activated within the context.
                      If False, this context manager has no effect.
            adapter_name: If specified, only this adapter will be enabled within the context,
                         and all other adapters will be disabled. If None, all adapters
                         retain their current state.
        """
        self.activated: bool = activated
        self.adapter_name = adapter_name
        
        # Early return if LoRA is not activated
        if not activated:
            return
        
        # Convert single module to list for uniform handling
        if not isinstance(modules, list) and not isinstance(modules, tuple):
            modules = [modules]
            
        # Filter only BaseTunerLayer modules (which implement LoRA adapters)
        self.lora_modules: List[BaseTunerLayer] = []
        for module in modules:
            if isinstance(module, BaseTunerLayer):
                self.lora_modules.append(module)
        
        # Store original state for all adapters to enable restoration when exiting the context
        self.original_active_adapters = {}
        self.scales = {}
        
        for i, lora_module in enumerate(self.lora_modules):
            # Store current active adapters
            self.original_active_adapters[i] = list(lora_module.active_adapters) if hasattr(lora_module, 'active_adapters') else []
            
            # Store current scaling values for all adapters
            self.scales[i] = {}
            for adapter in (lora_module.active_adapters if hasattr(lora_module, 'active_adapters') else []):
                self.scales[i][adapter] = lora_module.scaling.get(adapter, 1.0)
                
    def __enter__(self) -> None:
        """
        Enter the context and apply the specified LoRA configuration.
        
        This method:
        1. If adapter_name is specified, disables all adapters and enables only the specified one
        2. Otherwise, maintains the current adapter state but ensures all are properly scaled
        """
        if not self.activated or not self.lora_modules:
            return
            
        for i, lora_module in enumerate(self.lora_modules):
            # If adapter_name is specified, only enable that adapter
            if self.adapter_name is not None:
                # First, disable all active adapters by setting their scaling to 0.0
                for adapter in (lora_module.active_adapters if hasattr(lora_module, 'active_adapters') else []):
                    lora_module.scaling[adapter] = 0.0
                
                # Then enable only the specified adapter if it exists
                if hasattr(lora_module, 'adapters') and self.adapter_name in lora_module.adapters:
                    # Try to update active_adapters if it exists
                    current_adapters = list(lora_module.active_adapters) if hasattr(lora_module, 'active_adapters') else []
                    if self.adapter_name not in current_adapters:
                        try:
                            new_adapters = tuple(current_adapters + [self.adapter_name])
                            # Use the module's set_adapter method if available
                            if hasattr(lora_module, 'set_adapter'):
                                lora_module.set_adapter(new_adapters)
                            else:
                                # Directly set the attribute as a fallback
                                object.__setattr__(lora_module, 'active_adapters', new_adapters)
                        except:
                            # If we can't set it, log a warning and continue
                            print(f"Warning: Could not set active adapter for {lora_module}")
                            pass
                    
                    # Set its scaling to the original value or default to 1.0
                    lora_module.scaling[self.adapter_name] = self.scales[i].get(self.adapter_name, 1.0)
            else:
                # If no specific adapter is specified, restore all adapters to their original scaling
                for adapter in (lora_module.active_adapters if hasattr(lora_module, 'active_adapters') else []):
                    lora_module.scaling[adapter] = self.scales[i].get(adapter, 1.0)
                    
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """
        Exit the context and restore the original LoRA configuration.
        
        This method:
        1. Restores all original scaling factors
        2. Attempts to restore the original set of active adapters
        
        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised in the context
            exc_tb: Exception traceback if an exception was raised in the context
        """
        if not self.activated or not self.lora_modules:
            return
            
        for i, lora_module in enumerate(self.lora_modules):
            # Restore original scaling values regardless of whether we can restore active_adapters
            for adapter in (lora_module.active_adapters if hasattr(lora_module, 'active_adapters') else []):
                if adapter in self.scales[i]:
                    lora_module.scaling[adapter] = self.scales[i][adapter]
            
            # Try to restore original active adapters using various approaches
            try:
                original_adapters = tuple(self.original_active_adapters[i])
                # Use the set_adapter method if available
                if hasattr(lora_module, 'set_adapter'):
                    lora_module.set_adapter(original_adapters)
                elif hasattr(lora_module, '_active_adapter'):
                    # Some implementations use _active_adapter instead
                    object.__setattr__(lora_module, '_active_adapter', original_adapters)
                else:
                    # As a last resort, try direct attribute access
                    object.__setattr__(lora_module, 'active_adapters', original_adapters)
            except:
                # If we can't restore active_adapters, at least ensure the scaling is correct
                pass


class set_lora_scale:
    """
    Context manager that temporarily sets specific scaling factors for LoRA adapters.
    
    This class allows fine-grained control over adapter scaling, which affects the 
    contribution of LoRA parameters to the model output. Higher scale values increase
    the influence of the adapter.
    
    Typical use cases:
    1. Temporarily increasing/decreasing the influence of specific adapters
    2. Setting an adapter to zero scale to effectively disable it without removing it
    3. Evaluating the model with different adapter scaling configurations
    
    Example usage:
    ```python
    with set_lora_scale(model.modules, scale=0.5, adapter_name="adapter1"):
        # adapter1 will have 0.5 scaling during this block
        output = model(input)
    # Outside the block, original adapter scaling is restored
    ```
    """
    
    def __init__(
        self, 
        modules: Union[List[nn.Module], nn.Module], 
        scale: float,
        adapter_name: Optional[str] = None
    ) -> None:
        """
        Initialize the set_lora_scale context manager.
        
        Args:
            modules: A single module or list of modules to apply the scaling to.
                     Only modules that are instances of BaseTunerLayer will be affected.
            scale: The scaling factor to apply to the adapter(s).
            adapter_name: If specified, only this adapter's scaling will be modified.
                         If None, all active adapters will be affected.
        """
        # Convert single module to list for uniform handling
        if not isinstance(modules, list) and not isinstance(modules, tuple):
            modules = [modules]
            
        # Filter only BaseTunerLayer modules
        self.lora_modules: List[BaseTunerLayer] = []
        for module in modules:
            if isinstance(module, BaseTunerLayer):
                self.lora_modules.append(module)
        
        self.scale = scale
        self.adapter_name = adapter_name
        
        # Store original scaling values to enable restoration when exiting the context
        self.original_scales = {}
        
        for i, lora_module in enumerate(self.lora_modules):
            self.original_scales[i] = {}
            if adapter_name is not None:
                # Store only the specified adapter's scale
                if adapter_name in lora_module.scaling:
                    self.original_scales[i][adapter_name] = lora_module.scaling[adapter_name]
            else:
                # Store all adapters' scales
                for adapter in (lora_module.active_adapters if hasattr(lora_module, 'active_adapters') else []):
                    self.original_scales[i][adapter] = lora_module.scaling.get(adapter, 1.0)
                
    def __enter__(self) -> None:
        """
        Enter the context and apply the specified scaling configuration.
        
        This method:
        1. If adapter_name is specified, sets scaling for only that adapter
        2. Otherwise, sets scaling for all active adapters
        """
        if not self.lora_modules:
            return
            
        for i, lora_module in enumerate(self.lora_modules):
            if self.adapter_name is not None:
                # Set scale only for the specified adapter if it exists
                if hasattr(lora_module, 'adapters') and self.adapter_name in lora_module.adapters:
                    # Make sure the adapter is active before setting its scale
                    current_adapters = list(lora_module.active_adapters) if hasattr(lora_module, 'active_adapters') else []
                    if self.adapter_name not in current_adapters:
                        try:
                            new_adapters = tuple(current_adapters + [self.adapter_name])
                            # Use the set_adapter method if available
                            if hasattr(lora_module, 'set_adapter'):
                                lora_module.set_adapter(new_adapters)
                            else:
                                object.__setattr__(lora_module, 'active_adapters', new_adapters)
                        except:
                            # If we can't set it, continue with the adapters we have
                            pass
                    # Set the scale for the specified adapter
                    lora_module.scaling[self.adapter_name] = self.scale
            else:
                # Set scale for all active adapters
                for adapter in (lora_module.active_adapters if hasattr(lora_module, 'active_adapters') else []):
                    lora_module.scaling[adapter] = self.scale
                    
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """
        Exit the context and restore the original scaling configuration.
        
        This method restores all original scaling factors for affected adapters.
        
        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised in the context
            exc_tb: Exception traceback if an exception was raised in the context
        """
        if not self.lora_modules:
            return
            
        for i, lora_module in enumerate(self.lora_modules):
            # Restore original scaling values
            for adapter, scale in self.original_scales[i].items():
                lora_module.scaling[adapter] = scale