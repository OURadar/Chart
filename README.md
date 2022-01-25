# ARRC Radar Datasets

This repository contains the data description and a few examples of reading the data into the Python space.

## Data Format

Data are stored in a self-describubg Network Common Data Form [NetCDF].

### Arrays

| Field | Type | Description |
| --- | --- | --- |
| Azimuth | Float | Array of azimuth angles in degrees | 
| Elevation | Float | Array of elevation angles in degrees |
| Beamwidth | Float | Array of beamwidth in degrees |
| GateWidth | Float | Array of gatewidth in meters |
| Corrected_Intensity | Float | Product in described units (global attributes) |

The field `Corrected_Intensity` corresponds to the radar product that is stored a file. This field name can be retrieved from the global attribute `TypeName`, see below.

### Global Attributes

| Field | Type | Description |
| --- | --- | --- |
| TypeName | Char | Name of the product |
| DataType | Char | Should be RadialSet |
| Latitude | Double | Latitude of the radar coordinate in degrees |
| Longitude | Double | Longitude of the radar coordinate in degrees |
| Height | Float | Height of the radar in meters
| Time | Long | Seconds since Epoch |
| FractionalTime | Float | Fractional seconds of time |
| attributes | Char | Other attributes |


# Chart

A collection of notebooks to read and plot radar data.

![Figure](blob/PX-20170220-050706-E2.4-Z.png)

[NetCDF]: https://www.unidata.ucar.edu/software/netcdf/