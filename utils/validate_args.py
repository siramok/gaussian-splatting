import matplotlib.pyplot as plt
from argparse import ArgumentTypeError


def validate_colormaps(colormaps):
    validated_maps = []
    for cmap in colormaps.split(","):
        if cmap not in plt.colormaps():
            raise ValueError(f"Invalid colormap: {cmap}")
        validated_maps.append(cmap)

    if not validated_maps:
        print("No valid colormaps found")
        raise

    return validated_maps

def validate_opacitymaps(opacitymaps):
    validated_maps = []
    if opacitymaps:
        for cmap in opacitymaps.split(","):
            if not cmap:
                continue
            if cmap not in ["linear", "inv_linear", "constant0.01", "constant0.1", "random"]:
                continue
            validated_maps.append(cmap)

    return validated_maps


def validate_resolution(value):
    presets = {
        "low": 256,
        "medium": 512,
        "high": 1024,
        "extreme": 2048,
    }

    try:
        size = int(value)
        if size <= 0:
            raise ArgumentTypeError("Resolution must be a positive integer")
        return size
    except ValueError:
        lower_value = value.lower()
        if lower_value in presets:
            return presets[lower_value]
        raise ArgumentTypeError(
            f"Invalid resolution. Choose from {list(presets.keys())} or a positive integer"
        )


def validate_spacing(spacing):
    x, y, z = spacing.split(",")
    try:
        # Try to convert to integer first
        x = float(x)
        y = float(y)
        z = float(z)
        if x <= 0 or y <= 0 or z <= 0:
            raise ArgumentTypeError("All spacing dimensions must be positive")
        return (x, y, z)
    except ValueError:
        raise ArgumentTypeError(
            "Invalid spacing. Must be a comma-separated list of 3 positive numbers e.g 1,1,1"
        )

def validate_dropout(value):
    try:
        fvalue = float(value)
    except ValueError:
        raise ArgumentTypeError("Dropout must be a numeric value.")
    
    # If the value is less than 1, treat it as a dropout percentage else an exact number to keep
    if fvalue < 1:
        return fvalue
    else:
        return int(fvalue)