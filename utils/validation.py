"""
Shared validation utilities for input classes

"""


def validate_positive_float(value, param_name, units):
    """Validate that a parameter is a positive number

    Args:
        value: The value to validate
        param_name (str): Name of the parameter for error messages
        units (str): Physical units for error messages

    Returns:
        float: The validated value as a float

    Raises:
        TypeError: If value is not numeric
        ValueError: If value is not positive

    Examples:
        >>> validate_positive_float(1e-20, 'mass', 'eV')
        1e-20
        >>> validate_positive_float(-1, 'mass', 'eV')
        ValueError: mass must be positive...
        >>> validate_positive_float("text", 'mass', 'eV')
        TypeError: mass must be a number...
    """
    if value is None:
        raise TypeError(
            f"{param_name} cannot be None. "
            f"Expected a positive number in {units}."
        )

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"{param_name} must be a number, got {type(value).__name__}. "
            f"Expected a positive number in {units}.\n"
            f"Example: {param_name}=1.0"
        )

    if value_float <= 0:
        raise ValueError(
            f"{param_name} must be positive, got {value_float}. "
            f"Expected a positive number in {units}.\n"
            f"Example: {param_name}=1.0"
        )

    return value_float


def validate_non_negative_float(value, param_name, units):
    """Validate that a parameter is a non-negative number (>= 0)

    Args:
        value: The value to validate
        param_name (str): Name of the parameter for error messages
        units (str): Physical units for error messages

    Returns:
        float: The validated value as a float

    Raises:
        TypeError: If value is not numeric
        ValueError: If value is negative
    """
    if value is None:
        raise TypeError(
            f"{param_name} cannot be None. "
            f"Expected a non-negative number in {units}."
        )

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"{param_name} must be a number, got {type(value).__name__}. "
            f"Expected a non-negative number in {units}.\n"
            f"Example: {param_name}=1.0"
        )

    if value_float < 0:
        raise ValueError(
            f"{param_name} must be non-negative, got {value_float}. "
            f"Expected a non-negative number in {units}.\n"
            f"Example: {param_name}=1.0"
        )

    return value_float


def validate_float_in_range(value, param_name, units, min_val=None, max_val=None):
    """Validate that a parameter is a number within a specified range

    Args:
        value: The value to validate
        param_name (str): Name of the parameter for error messages
        units (str): Physical units for error messages
        min_val (float, optional): Minimum allowed value (inclusive)
        max_val (float, optional): Maximum allowed value (inclusive)

    Returns:
        float: The validated value as a float

    Raises:
        TypeError: If value is not numeric
        ValueError: If value is outside the specified range
    """
    if value is None:
        raise TypeError(
            f"{param_name} cannot be None. "
            f"Expected a number in {units}."
        )

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"{param_name} must be a number, got {type(value).__name__}. "
            f"Expected a number in {units}.\n"
            f"Example: {param_name}=1.0"
        )

    if min_val is not None and value_float < min_val:
        raise ValueError(
            f"{param_name} must be >= {min_val}, got {value_float}. "
            f"Expected a number in {units}.\n"
            f"Example: {param_name}={min_val}"
        )

    if max_val is not None and value_float > max_val:
        raise ValueError(
            f"{param_name} must be <= {max_val}, got {value_float}. "
            f"Expected a number in {units}.\n"
            f"Example: {param_name}={max_val}"
        )

    return value_float


def validate_choice(value, param_name, valid_options):
    """Validate that a string parameter is one of the valid choices

    Args:
        value: The value to validate
        param_name (str): Name of the parameter for error messages
        valid_options (set or list): Valid choices for the parameter

    Returns:
        str: The validated value

    Raises:
        TypeError: If value is not a string
        ValueError: If value is not in valid_options

    Examples:
        >>> validate_choice('scalar', 'ULB_type', {'scalar', 'ALP'})
        'scalar'
        >>> validate_choice('electron', 'coupling_type', {'electron', 'photon'})
        'electron'
    """
    if not isinstance(value, str):
        valid_list = sorted(list(valid_options)) if isinstance(valid_options, set) else valid_options
        raise TypeError(
            f"{param_name} must be a string, got {type(value).__name__}.\n"
            f"Valid options: {valid_options}\n"
            f"Example: {param_name}='{valid_list[0]}'"
        )

    if value not in valid_options:
        valid_list = sorted(list(valid_options)) if isinstance(valid_options, set) else valid_options
        raise ValueError(
            f"{param_name} '{value}' is not valid.\n"
            f"Valid options: {valid_options}\n"
            f"Example: {param_name}='{valid_list[0]}'"
        )

    return value


def validate_dict_structure(value, param_name, key_type=str, value_type=float,
                           allow_empty=False, value_validator=None):
    """Validate dictionary structure and contents

    Args:
        value: The dictionary to validate
        param_name (str): Name of the parameter for error messages
        key_type (type): Expected type for dictionary keys
        value_type (type): Expected type for dictionary values
        allow_empty (bool): Whether to allow empty dictionaries
        value_validator (callable, optional): Function to validate each value

    Returns:
        dict: The validated dictionary (with values possibly converted/validated)

    Raises:
        TypeError: If not a dictionary or types are wrong
        ValueError: If dictionary is empty (when not allowed) or validation fails
    """
    if not isinstance(value, dict):
        raise TypeError(
            f"{param_name} must be a dictionary, got {type(value).__name__}.\n"
            f"Example: {param_name}={{'key1': 1.0, 'key2': 2.0}}"
        )

    if not allow_empty and len(value) == 0:
        raise ValueError(
            f"{param_name} dictionary cannot be empty.\n"
            f"Example: {param_name}={{'key1': 1.0, 'key2': 2.0}}"
        )

    # Validate each entry
    validated = {}
    for k, v in value.items():
        if not isinstance(k, key_type):
            raise TypeError(
                f"{param_name} keys must be {key_type.__name__}, "
                f"got {type(k).__name__} for key {k}.\n"
                f"Example: {param_name}={{'key1': 1.0, 'key2': 2.0}}"
            )

        try:
            v_converted = value_type(v)
        except (TypeError, ValueError):
            raise TypeError(
                f"{param_name}['{k}'] must be {value_type.__name__}, "
                f"got {type(v).__name__}.\n"
                f"Example: {param_name}={{'{k}': 1.0}}"
            )

        # Apply additional validation if provided
        if value_validator is not None:
            v_converted = value_validator(v_converted, f"{param_name}['{k}']")

        validated[k] = v_converted

    return validated


def validate_positive_dict_values(value, param_name):
    """Helper validator for dictionaries with positive numeric values"""
    def validate_value(v, vname):
        if v <= 0:
            raise ValueError(
                f"{vname} must be positive, got {v}.\n"
                f"Expected positive numeric value."
            )
        return v

    return validate_dict_structure(
        value, param_name,
        key_type=str,
        value_type=float,
        allow_empty=False,
        value_validator=validate_value
    )
