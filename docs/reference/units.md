# Spatial units

NeuRosetta tracks spatial scale in ``tree.metadata["units"]``. Values are normalized
through [Pint](https://pint.readthedocs.io/) where possible. Legacy strings such as
``undefined`` map to Pint's ``dimensionless`` default.

## Supported units and aliases

| Canonical | Aliases | Notes |
| --- | --- | --- |
| `dimensionless` | `dimensionless`, `undefined`, `Undefined`, `1 dimensionless` | Default when no spatial scale is assigned. |
| `nanometer` | `nm`, `nanometer` | Common for EM reconstructions. |
| `micron` | `micron`, `micrometer`, `um`, `µm`, `microns` | Default harmonization target for forests. |
| `millimeter` | `mm`, `millimeter` | Standard SI length unit. |
| `meter` | `m`, `meter` | Standard SI length unit. |
| `voxel` | `voxel` | Voxel-index coordinates; see below. |

Other SI length units accepted by Pint (for example `angstrom`) can also be used,
but only the aliases above are guaranteed stable in metadata normalization.

## Voxel coordinates

Voxel units represent coordinates stored as **voxel indices**, where each voxel is a
cube with a known edge length. Set them with a size and base unit:

```python
import NeuRosetta as nr

tree.set_voxel_units(8, "nm")          # 8 nm cubic voxels
tree.set_units("voxel", voxel_size=8, voxel_unit="nm")  # equivalent

tree.get_voxel_spec()                  # -> (8.0, "nanometer")
tree.convert_units("micron")           # rescale geometry to microns
```

Required metadata when ``units == "voxel"``:

| Key | Type | Meaning |
| --- | --- | --- |
| `voxel_size` | `float` | Edge length of one voxel |
| `voxel_unit` | `str` | Spatial unit for that edge length |

On import:

```python
tree = nr.import_swc("42.swc", set_units="voxel", voxel_size=8, voxel_unit="nm")
```

SWC headers round-trip voxel metadata:

```text
# Meta: {"id": "1", "units": "voxel", "voxel_size": 8.0, "voxel_unit": "nanometer"}
```

## Programmatic reference

```python
import NeuRosetta as nr

for definition in nr.list_unit_definitions():
    print(definition.canonical, definition.aliases)

print(nr.format_units_reference_table())
```
