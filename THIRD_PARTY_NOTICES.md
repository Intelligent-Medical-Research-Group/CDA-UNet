# Scientific attribution and third-party notices

## UNITO and the reference dataset

CDA-UNet builds on the bivariate density-image gating formulation described in:

> Chen, J., Ionita, M., Feng, Y., et al. (2025). Automated cytometric gating with
> human-level performance using bivariate segmentation. *Nature Communications*,
> 16, 1576. https://doi.org/10.1038/s41467-025-56622-2

The original UNITO software is maintained at
[KyleeCJ/UNITO](https://github.com/KyleeCJ/UNITO). Its copyright notice identifies
the Trustees of the University of Pennsylvania (2022), and its
[upstream license](https://github.com/KyleeCJ/UNITO/blob/main/license) restricts
redistribution of the original software and modifications. That software is not
vendored in this repository. Attribution to the paper does not grant permission
to redistribute upstream source or weights. This repository does not sublicense
UNITO or change its terms.

The CDA-UNet package provides a newly written implementation of its density-image
pipeline and conditional dynamic adapter model. Scientific descriptions, data
schemas, and architecture choices are attributed to the relevant publications;
the original UNITO application, notebooks, utilities, and graphical interface
are not included.

The public data used in the study can be obtained from
[Pennsieve Discover, dataset 408, version 2](https://discover.pennsieve.io/datasets/408/version/2).
Download and use the data under the dataset's own published terms. This
repository does not redistribute measurements, expert annotations, trained
weights, or benchmark prediction files. Citation and availability of the
dataset should not be interpreted as ownership of those materials.

## Other scientific foundations

- U-Net: Ronneberger, O., Fischer, P., and Brox, T. (2015). *U-Net:
  Convolutional Networks for Biomedical Image Segmentation*.
  https://doi.org/10.1007/978-3-319-24574-4_28
- Batch normalization: Ioffe, S., and Szegedy, C. (2015). *Batch Normalization:
  Accelerating Deep Network Training by Reducing Internal Covariate Shift*.
  https://proceedings.mlr.press/v37/ioffe15.html
- Group normalization: Wu, Y., and He, K. (2018). *Group Normalization*.
  https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html

These citations acknowledge methods. No implementation from these papers is
vendored here.

## Runtime dependencies

PyTorch, NumPy, SciPy, pandas, OpenCV, and the other packages listed in the
installation metadata are installed separately. They retain their own copyright
and license terms. Refer to each installed distribution's license files when
redistributing an environment or container. A license for this repository,
if added by its rights holders, applies only to materials those holders are
authorized to license.
