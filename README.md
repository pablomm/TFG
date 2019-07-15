# *Functional data analysis*: interpolation, registration, and nearest neighbors in _scikit-fda_

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Bachelor's thesis to obtain a double degree in Computer Science and Mathematics at the Autonomous University of Madrid.
This repository contains the files used to generate the undergraduate thesis 
[document](https://github.com/pablomm/TFG/blob/master/tfg-pablo-marcos.pdf). 
The contributions made in this work can be found in the [_scikit-fda_](https://github.com/GAA-UAM/)  project repository.

## Abstract

Functional Data Analysis (**FDA**) is a branch of Statistics devoted
to the study of random quantities that depend on a continuous parameter,
such as time series or curves in space.
In FDA the data instances can be viewed as random functions
sampled from an underlying stochastic process.

In this work we consider three different tasks in FDA:
the use of interpolation techniques to estimate the values
of the functions at unobserved points,
the registration of these type of data,
and the solution of classification and regression problems in which
the instances are characterized by functional attributes.
In particular, in this project the _scikit-fda_ package
for FDA in Python has been extended
with functionality in these areas.

Generally, the data instances considered in FDA consist of a
collection of observations at a discrete values of the parameter
on which they depend (e.g. time or space).
For some applications it is convenient, and in some cases
necessary, to estimate the value of these functions
at unobserved points.
This can be achieved through the use of interpolation
from the available measurements.

In some applications, the functions observed
have similar shapes, but exhibit variability whose
origin can be traced to distortions in the scale
of the continuous parameter on which the data depend.
Registration consists in characterizing
this variability and eliminating it from the sample considered.

In this work we also address classification and regression problems
with data that are characterized by functions.
Specifically, we design nearest neighbors estimators
based on the notion of closeness among samples.

Specifically, in this work
the _scikit-fda_ package has been extended
to include interpolation methods based on splines.
The package has also been endowed with tools
for data registration using either shifts,
landmark alignment, or elastic registration,
which makes use of the Fisher-Rao metric
to align the functions in a sample.
In addition, models based on nearest neighbors
have been included to carry out regression, with both scalar and functional
response, and classification.
