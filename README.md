# CRB
Used to calculate Cramer Rao Lower bounds for complex gain uncertainties in
radio interferometers, due to pointing, number of tiles/stations, field of view,
and other factors.

> [!WARNING]
> Assumptions are made in this calculation:
> * All pointing are treated as if they are at zenith. I don't do any fancy
>   precession or coordinate projection.
> * All stations are identical
> * Only works on sources within field of view of station (station diameter /
>   wavelength)

# Installation
```
cargo install --path . --locked
```

If you would like to use the OpenCL capability on Apple silicone macs, then you
need to compile with the `gpu-single` feature, which does the Fisher Information
Matrix calculation with single point precision.

```
cargo install --path . --locked --features=gpu-single
```

# Running
Requires a mwa_hyperdrive compliant source list in yaml format.

You'll need to create a config toml, here's an example with all the needed
options defined:
```{toml}
use_gpu = true
srclist = "/path/to/srclist.yaml"
output = "./output"

[obs_config]
obs_name = "EoR0"
ra = 0
dec = -27
channel_width = 80e3
start_freq = 182e6
end_freq = 197e6
int_time = 8
t_sys = 200

[tel_config]
telescope = "ska"
station_diameter = 18
telescope_layout_file = "/path/to/array_layout_file"
station_layout_file = "/path/to/station_layout_file"
```

The `telescope_layout_file` should be a CSV file with local (x, y) coordinates
of each station relative to centre of array. The `station_layout_file` should be
in similar format, but coordinates are relative to centre of station.
