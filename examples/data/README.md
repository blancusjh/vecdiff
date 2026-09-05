# Optical prescription data

`US7557996.csv` is the native CSV export of US7557996B2, Fig. 3 / Table 3,
retrieved from `blancusjh/geometrical-raytracer` revision
`87d2e66a10c18456b590a170c0d1b460fae9d4ac`, path
`data/optical_systems/lithography/US7557996_Fig3_Table3_prescription.csv`.

Lengths are millimetres; the index column is tabulated at 193.368 nm. All 48
encounters are retained, including two mirrors and a stop. This is a geometry
and IO fixture, not evidence that the full objective has been wave-propagated.

## DUV pupil wavefront reference

`duv_wavefront.json` is the unmodified `vectorwave/data/duv.json` from
[blancusjh/wavec at dc74da5ff9c283a0d7b4ef85febc8ead4288e2d3](https://github.com/blancusjh/wavec/blob/dc74da5ff9c283a0d7b4ef85febc8ead4288e2d3/vectorwave/data/duv.json).
It describes the US7557996 Fig. 3/Table 3 DUV objective at 193.368 nm:
NA 1.2, image index 1.59667693, reduction 4:1, and a 129×129 phase grid in
vacuum-wavelength optical-path units. The map covers normalized pupil
coordinates [-1,1]², rows y / columns x, and was generated at the 62 mm chief
object field. Zeros outside the unit disk are masked, not physical measurements.

The sixth notebook uses this dataset in the independent pupil reference only.
It does not infer coatings, pupil amplitude, full-field behavior, or spectral
propagation through the accompanying prescription. The main library never
loads it. JSON inputs participate in the executed-notebook fingerprint.
