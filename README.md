# ARRC Mobile Radar Datasets

This repository contains the data description and a few examples of reading the data into a Python and MATLAB space.

## Data Format

Data are stored in a self-describubg Network Common Data Form [NetCDF].

### Arrays

| Field | Type |
| --- | --- |
| Azimuth | Float |
| Elevation | Float |
| Beamwidth | Float |
| GateWidth | Float |
| Corrected_Intensity | Float |

The field `Corrected_Intensity` corresponds to the radar product that is stored a file. This field name can be retrieved from the global attribute `TypeName`, see below.

### Global Attributes

| Field | Type |
| --- | --- |
| TypeName | Char |
| DataType | Char |
| Latitude | Double |
| Longitude | Double |
| Height | Float |
| Time | Long |
| FractionalTime | Float |
| attributes | Char |


# Chart

A collection of notebooks to read and plot radar data.

![Figure](blob/PX-20170220-050706-E2.4-Z.png)

[NetCDF]: https://www.unidata.ucar.edu/software/netcdf/