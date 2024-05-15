#include "mex.h"

#if ! defined (MX_API_VER) | (MX_API_VER < 0x07030000)
typedef unsigned int mwSize ;
#endif

#include <stdlib.h>

void mexFunction(
    int		  nout, 	/* number of expected outputs */
    mxArray	  *out[],	/* mxArray output pointer array */
    int		  nin, 	/* number of inputs */
    const mxArray	  *in[]	/* mxArray input pointer array */
    )
{
   
  enum {IN_MAP=0,IN_NLABELS} ;
  enum {OUT_NEIGHBORS} ;

  int nlabels, rows, cols;
  double * map, * neighbors;
  int i, c, r;
  mwSize dims[2];

  /****************************************************************************
   * ERROR CHECKING
   ***************************************************************************/
  if (nin != 2)
    mexErrMsgTxt("Two arguments are required.");
  if (nout > 1)
    mexErrMsgTxt("Only one output argument allowed");

  if(mxGetClassID(in[IN_MAP]) != mxDOUBLE_CLASS ||
     mxGetNumberOfDimensions(in[IN_MAP]) != 2) 
    mexErrMsgTxt("MAP must be a 2D matrix of doubles");

  if(mxGetClassID(in[IN_NLABELS]) != mxDOUBLE_CLASS)
    mexErrMsgTxt("NLABELS must be a double");

  nlabels = (int) *mxGetPr(in[IN_NLABELS]);
  rows = mxGetM(in[IN_MAP]);
  cols = mxGetN(in[IN_MAP]);

  dims[0] = nlabels;
  dims[1] = nlabels;

  out[OUT_NEIGHBORS] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);

  map = mxGetPr(in[IN_MAP]);
  neighbors = mxGetPr(out[OUT_NEIGHBORS]);

  i = 0;
  for (c = 0; c < cols; c++ )
    for (r = 0; r < rows; r++ )
    {
      if(r != rows-1)
      {
        int l1 = (int)map[i]-1;
        int l2 = (int)map[i+1]-1;
        if(l1 != l2) 
        {
          neighbors[ l1 + l2*nlabels ]++;
          neighbors[ l2 + l1*nlabels ]++;
        }
      }
      if(c != cols-1)
      {
        int l1 = (int)map[i]-1;
        int l2 = (int)map[i+rows]-1;
        if(l1 != l2) 
        {
          neighbors[ l1 + l2*nlabels ]++;
          neighbors[ l2 + l1*nlabels ]++;
        }
      }
      i++;
    } 
}
