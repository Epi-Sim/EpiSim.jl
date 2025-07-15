"""
Dynamic JSON Schema Validator for EpiSim Configuration

This module provides JSON schema validation with dynamic group size handling.
The schema template is loaded and group-dependent array sizes are populated
based on the actual G_labels configuration.
"""

import json
import os
from typing import Dict, Any, List, Optional
import jsonschema
from jsonschema import ValidationError


class SchemaValidator:
    """
    Dynamic JSON schema validator for EpiSim configurations.
    
    This class loads a schema template and dynamically generates schemas
    based on the actual group size in the configuration.
    """
    
    def __init__(self, schema_template_path: Optional[str] = None):
        """
        Initialize the schema validator.
        
        Args:
            schema_template_path: Path to the schema template JSON file.
                                If None, uses the default template.
        """
        if schema_template_path is None:
            # Use default template in the same directory as this file
            schema_template_path = os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "episim_schema_template.json"
            )
        
        self.schema_template_path = os.path.abspath(schema_template_path)
        self._load_template()
    
    def _load_template(self):
        """Load the schema template from file."""
        try:
            with open(self.schema_template_path, 'r', encoding='utf-8') as f:
                self.schema_template = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Schema template not found at: {self.schema_template_path}"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in schema template: {e}"
            )
    
    def generate_schema(self, group_size: int) -> Dict[str, Any]:
        """
        Generate a complete schema with dynamic group sizes.
        
        Args:
            group_size: Number of demographic groups (length of G_labels)
            
        Returns:
            Complete JSON schema with resolved group sizes
        """
        if group_size < 1:
            raise ValueError("Group size must be at least 1")
        
        # Convert template to string, replace placeholders, then parse back
        schema_str = json.dumps(self.schema_template, indent=2)
        schema_str = schema_str.replace('"{GROUP_SIZE}"', str(group_size))
        
        return json.loads(schema_str)
    
    def validate(self, config: Dict[str, Any], verbose: bool = True) -> List[str]:
        """
        Validate a configuration against the dynamic schema.
        
        Args:
            config: Configuration dictionary to validate
            verbose: Whether to print detailed error messages
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            # Extract group size from configuration
            group_size = self._extract_group_size(config)
            
            # Generate schema with correct group size
            schema = self.generate_schema(group_size)
            
            # Validate against schema
            jsonschema.validate(config, schema)
            
            if verbose:
                print(f"JSON Schema validation passed (group size: {group_size})")
            
        except ValidationError as e:
            error_msg = self._format_validation_error(e)
            errors.append(error_msg)
            if verbose:
                print(f"JSON Schema validation failed: {error_msg}")
        
        except Exception as e:
            error_msg = f"Schema validation error: {str(e)}"
            errors.append(error_msg)
            if verbose:
                print(error_msg)
        
        return errors
    
    def _extract_group_size(self, config: Dict[str, Any]) -> int:
        """
        Extract the group size from the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Number of demographic groups
            
        Raises:
            ValueError: If G_labels is missing or invalid
        """
        try:
            g_labels = config["population_params"]["G_labels"]
            if not isinstance(g_labels, list) or len(g_labels) < 1:
                raise ValueError("G_labels must be a non-empty list")
            return len(g_labels)
        except KeyError:
            raise ValueError("Missing required field: population_params.G_labels")
        except TypeError:
            raise ValueError("Invalid configuration structure")
    
    def _format_validation_error(self, error: ValidationError) -> str:
        """
        Format a JSON schema validation error into a readable message.
        
        Args:
            error: ValidationError from jsonschema
            
        Returns:
            Formatted error message
        """
        if error.absolute_path:
            path = ".".join(str(p) for p in error.absolute_path)
            return f"Validation error at '{path}': {error.message}"
        else:
            return f"Validation error: {error.message}"


class EpiSimSchemaValidator:
    """
    Convenience wrapper for EpiSim-specific schema validation.
    """
    
    def __init__(self):
        self.validator = SchemaValidator()
    
    def validate_config(self, config: Dict[str, Any], verbose: bool = True) -> bool:
        """
        Validate an EpiSim configuration and raise exception if invalid.
        
        Args:
            config: Configuration dictionary
            verbose: Whether to print validation messages
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        errors = self.validator.validate(config, verbose)
        
        if errors:
            error_msg = f"Configuration validation failed with {len(errors)} error(s):\n"
            for i, err in enumerate(errors, 1):
                error_msg += f"  {i}. {err}\n"
            raise ValueError(error_msg.strip())
        
        return True
    
    def validate_config_safe(self, config: Dict[str, Any], verbose: bool = True) -> tuple[bool, List[str]]:
        """
        Validate an EpiSim configuration without raising exceptions.
        
        Args:
            config: Configuration dictionary
            verbose: Whether to print validation messages
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = self.validator.validate(config, verbose)
        return len(errors) == 0, errors
    
    def get_schema_for_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the complete schema for a given configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Complete JSON schema
        """
        group_size = self.validator._extract_group_size(config)
        return self.validator.generate_schema(group_size)


# Convenience functions for backward compatibility
def validate_episim_config(config: Dict[str, Any], verbose: bool = True) -> bool:
    """
    Validate an EpiSim configuration using JSON schema.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print validation messages
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    validator = EpiSimSchemaValidator()
    return validator.validate_config(config, verbose)


def validate_episim_config_safe(config: Dict[str, Any], verbose: bool = True) -> tuple[bool, List[str]]:
    """
    Validate an EpiSim configuration without raising exceptions.
    
    Args:
        config: Configuration dictionary
        verbose: Whether to print validation messages
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = EpiSimSchemaValidator()
    return validator.validate_config_safe(config, verbose)