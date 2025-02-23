# Non-Covalent Interactions Atlas

Non-Covalent Interactions Atlas is a collection of computational chemistry data sets of accurate interaction energies in non-covalent complexes covering different classes of interactions in an extended chemical space.

More information is available at the [NCIAtlas website](http://www.nciatlas.org).

## Repository oganization

The core part are the geometries of the systems, in the widespread .xyz format. Each geometry also includes a header with all the information necessary for performing the calculations and the benchmark interaction energy. For many applications, these files are all what is needed. The geometries for each data set are in subdirectories of the [geometries directory](https://github.com/Honza-R/NCIAtlas/tree/main/geometries).

Additional data including components of the benchmark interaction energy and results of other calculations featured in the publications are provided as plain text tables in the [tables directory](https://github.com/Honza-R/NCIAtlas/tree/main/tables). These are also packaged together with geometries in the [packaged_sets directory](https://github.com/Honza-R/NCIAtlas/tree/main/packaged_sets).

All these data are also provided in a structured YAML file used by the [Cuby framework](http://cuby4.molecular.cz/). These files are located in the [dataset_definition_files directory](https://github.com/Honza-R/NCIAtlas/tree/main/dataset_definition_files).

## Data versions

The data provided here are available also at the [NCIAtlas website](http://www.nciatlas.org) and as a supporting information of the individual papers. The data (molecular geometries, computed results) should be the same, but their organization, file names, etc. may differ. This repository aims to provide the data in a structure as consistent as possible across all the data sets.

## References

### D1200 - London dispersion in an extended chemical space
[J. Řezáč, ChemRxiv preprint, 2022](https://doi.org/10.26434/chemrxiv-2022-pl3r8)

### D442x10 - London dispersion in an extended chemical space, 10-point dissociation curves
[J. Řezáč, ChemRxiv preprint, 2022](https://doi.org/10.26434/chemrxiv-2022-pl3r8)

### SH250x10 - Sigma-hole interactions
[K. Kříž, J. Řezáč, ChemRxiv preprint, 2022](https://doi.org/10.26434/chemrxiv-2022-x72mz)

### R739×5 - Repulsive contacts in an extended chemical space
[K. Kříž, M. Nováček, J. Řezáč, J. Chem. Theory Comput., 2021, 17, 1548–1561.](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c01341)

### HB300SPX×10 - Hydrogen bonding extended to S, P and halogens
[J. Řezáč, J. Chem. Theory Comput., 2020, 16, 6305–6316.](https://dx.doi.org/10.1021/acs.jctc.0c00715)

### HB375×10 - Hydrogen bonding in organic molecules
[J. Řezáč, J. Chem. Theory Comput., 2020, 16, 2355–2368.](https://doi.org/10.1021/acs.jctc.9b01265)

### IHB100×10 - Ionic hydrogen bonds in organic molecules
[J. Řezáč, J. Chem. Theory Comput., 2020, 16, 2355–2368.](https://doi.org/10.1021/acs.jctc.9b01265)

