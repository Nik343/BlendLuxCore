"""Various utilities requiring pyluxcore."""

import pyluxcore
from functools import lru_cache



def create_props(prefix, definitions):
    """
    :param prefix: string, will be prepended to each key part of the definitions.
                   Example: "scene.camera." (note the trailing dot)
    :param definitions: dictionary of definition pairs. Example: {"fieldofview", 45}
    :return: pyluxcore.Properties() object, initialized with the given definitions.
    """
    props = pyluxcore.Properties()

    for key, value in definitions.items():
        props.Set(pyluxcore.Property(prefix + key, value))

    return props


def matrix_to_list(matrix, invert=False):
    """Flatten a 4x4 matrix into a list

    Returns list[16]
    """
    # Copy required for BlenderMatrix4x4ToList(), not sure why, but if we don't
    # make a copy, we only get an identity matrix in C++
    matrix = matrix.copy()

    if invert:
        matrix.invert_safe()

    return pyluxcore.BlenderMatrix4x4ToList(matrix)


def is_opencl_build():
    """Check if pyluxcore has been built with OpenCL support."""
    return (
        pyluxcore.GetPlatformDesc()
        .Get("compile.LUXRAYS_ENABLE_OPENCL")
        .GetBool()
    )


def is_cuda_build():
    """Check if pyluxcore has been built with Cuda support."""
    return (
        pyluxcore.GetPlatformDesc()
        .Get("compile.LUXRAYS_ENABLE_CUDA")
        .GetBool()
    )

@lru_cache(maxsize=None)
def _get_render_engine_names():
    names = set()
    getters = (
        getattr(pyluxcore, "GetRenderEnginePluginNames", None),
        getattr(pyluxcore, "GetRenderEngineNames", None),
    )
    for getter in getters:
        if getter is None:
            continue
        try:
            result = getter()
        except Exception:
            continue
        names.update(_normalize_engine_names(result))

    registry_entry = getattr(pyluxcore, "RenderEngineRegistry", None)
    if registry_entry is not None:
        try:
            registry = registry_entry() if callable(registry_entry) else registry_entry
            get_names = getattr(registry, "GetNames", None)
            if callable(get_names):
                names.update(_normalize_engine_names(get_names()))
        except Exception:
            pass

    return tuple(sorted(names))


def _normalize_engine_names(raw):
    if isinstance(raw, str):
        return {raw.upper()}
    try:
        iterator = iter(raw)
    except TypeError:
        return {str(raw).upper()}
    return {str(name).upper() for name in iterator}


def is_bidir_cuda_supported():
    if not is_cuda_build():
        return False
    return "BIDIRCUDA" in _get_render_engine_names()


