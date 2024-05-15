
SMALLbox Version 2.0

25th May, 2012

Version 2.0 of SMALLbox is the release candidate that is distributed 
for testing and bug fixing purposes. Please send all bugs, requests and suggestions to:

luis.figueira@soundsoftware.ac.uk

---------------------------------------------------------------------------

Copyright (2012): 	Luis Figueira, Ivan Damnjanovic, Matthew Davies
			Centre for Digital Music, 
			Queen Mary University of London

SMALLbox is distributed under the terms of the GNU General Public License 3

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

---------------------------------------------------------------------------

SMALLbox is a MATLAB Evaluation Framework for the EU FET-OPEN Project, no: 225913:
Sparse Models, Algorithms, and Learning for Large-scale data (SMALL). 

The project team includes researchers from the following institutions:

INRIA:                            http://www.irisa.fr/metiss/gribonval/
University of Edinburgh:          http://www.see.ed.ac.uk/~mdavies4/
Queen Mary, University of London: http://www.elec.qmul.ac.uk/people/markp/
EPFL:                             http://people.epfl.ch/pierre.vandergheynst
Technion:                         http://www.cs.technion.ac.il/~elad/

---------------------------------------------------------------------------

will download and install the following existing toolboxes 
related to Sparse Representations, Compressed Sensing and Dictionary Learning:


-	SPARCO (v.1.2) - set of sparse representation problems[5]
-	SparseLab (v.2.1) - set of sparse solvers [1]
-	Sparsify (v.0.4) - set of greedy and hard thresholding algorithms [2]
-	SPGL1 (v.1.7) - large-scale sparse reconstruction solver [3]
-	GPSR (v.6.0) - Gradient projection for sparse reconstruction [4]
-	KSVD-box (v.13) and OMP-box (v.10) - dictionary learning [6]
-	KSVDS-box (v.11) and OMPS-box (v.1) - sparse dictionary learning [7]. 



IMPORTANT:
In order to use SparseLab please register at
http://cgi.stanford.edu/group/sparselab/cgi-bin/register.pl

IMPORTANT:
To successfully install all toolboxes you will need to have MEX setup to compile C files.
If this is not already setup, run "mex -setup" or type "help mex" in the MATLAB command prompt.

IMPORTANT:
Because the toolboxes are downloaded automatically, you must have an internet connection
to successfully install SMALLbox.

IMPORTANT:
If you are running Matlab on MAC OSX or Linux, you must start Matlab with the jvm enabled.
Not doing so, will prevent you being able to unzip the downloaded toolboxes.

To install the toolbox run the command "SMALLboxsetup" from the MATLAB command prompt.

Once installed, there are two optional demo functions that can be run:

Example test of solvers from different toolboxes on Sparco compressed sensing problems and
Example test of dictionary learning techniques on image denoising problems

Further examples can be found in {SMALLbox root}/examples directory

---------------------------------------------------------------------------

For more information on the SMALL Project, please visit the following website:

http://small-project.eu


Contact: luis.figueira@soundsoftware.ac.uk

This code is in experimental stage; any comments or bug reports are 
very welcome. More information about using SMALLbox will be includied in release version 
documenation file.

References:

1.	Donoho, D., Stodden, V., Tsaig, Y.: Sparselab. 2007, http://sparselab.stanford.edu/
2.	Blumensath, T., Davies, M. E.: Gradient pursuits. 
	In IEEE Transactions on Signal Processing, vol. 56, no. 6, pp. 2370–2382, June 2008.
3.	Berg, E. v., Friedlander, M. P.: Probing the pareto frontier for basis pursuit solutions. 
	In SIAM Journal on Scientific Computing, vol. 31, no. 2, pp. 890–912, 2008.
4.	Figueiredo, M. A. T., Nowak, R. D., Wright, S. J.: Gradient projection for sparse
	reconstruction: Application to compressed sensing and other inverse problems. In Journal 
	of Selected Topics in Signal Processing:Special Issue on Convex Optimization for Signal Processing, December 2007.
5.	Berg, E. v., Friedlander, M. P., Hennenfent, G., Herrmann, F., Saab, R., Yilmaz, O.: Sparco:
	A testing framework for sparse reconstruction. In ACM Trans. on Mathematical Software, 35(4):1-16, February 2009.
6.	Aharon, M., Elad, M., Bruckstein, A. M.: The K-SVD: An algorithm for designing of overcomplete
	dictionaries for sparse representation. In IEEE Transactions on Signal Processing, vol. 54, no. 11, pp. 4311–4322, 2006.
7.	Rubinstein, R., Zibulevsky, M. and Elad, M.: Double Sparsity: Learning Sparse Dictionaries 
	for Sparse Signal Approximation. In IEEE Transactions on Signal Processing, Vol. 58, No. 3, Pages 1553-1564, March 2010.
