#include "stdafx.h"
#include "KDSearch.h"

#include "PKDTree.h"
#include "TaskManager.h"

namespace matlab {
#include "mex.h"
}

using namespace matlab;


class SearchNNTask : public ANNThreading::ITask
{
private:
	int m_indexStart;
	int m_indexEnd;
	PKDTree& m_kdtree;
	const Dataset<float>& m_testset;
	Dataset<int>& m_result;
	Dataset<float>& m_dists;
	float m_epsilon;
public:

	SearchNNTask(int indexStart, int indexEnd, PKDTree& kdtree, const Dataset<float>& testset, Dataset<int>& result,  Dataset<float>& dists, float epsilon)
		: m_indexStart(indexStart),
		m_indexEnd(indexEnd),
		m_kdtree(kdtree),
		m_testset(testset),
		m_result(result),
		m_dists(dists),
		m_epsilon(epsilon)
	{
	}


	virtual void Run()
	{
	   int nn = m_result.cols;

		KNNResultSet resultSet(nn);

		for (int i = m_indexStart; i < m_indexEnd && i < m_testset.rows; i++) 
		{
			float* target = m_testset[i];
			resultSet.init(target, m_testset.cols);

			m_kdtree.getNeighbors(resultSet,target, m_epsilon);

			int* neighbors = resultSet.getNeighbors();
			float* distances = resultSet.getDistances();
			memcpy(m_result[i], neighbors, nn*sizeof(int));
			memcpy(m_dists[i], distances, nn*sizeof(float));
		}
	}
};

void search_for_neighbors(PKDTree& kdtree, const Dataset<float>& testset, Dataset<int>& result,  Dataset<float>& dists, float epsilon)
{
    //assert(testset.rows == result.rows);

    int nn = result.cols;

	KNNResultSet resultSet(nn);

//	SearchNNTask task(0, testset.rows, kdtree, testset, result, dists, epsilon);
//	task.Run();

	ANNThreading::TaskManager tm;
	int block_size = testset.rows / (4*tm.GetNumberOfThreads());
	if(block_size < 100)block_size = 100;

	int nBlocks = (int)(testset.rows/(float)block_size) + 1;

	for(int i = 0 ; i < nBlocks ; i++)
	{
		tm.ScheduleTask(new SearchNNTask(i*block_size, (i+1)*block_size,
			kdtree, testset, result, dists, epsilon));
	}
	tm.WaitAll();
}


void flann_find_nearest_neighbors(float* dataset,  int rows, int cols, float* testset, int tcount, int* result, float* dists, int nn, float epsilon)
{
	Dataset<float> inputData(rows,cols,dataset);
	PKDTree kdtree(inputData);

	Dataset<int> result_set(tcount, nn, result);
	Dataset<float> dists_set(tcount, nn, dists);
	search_for_neighbors(kdtree, Dataset<float>(tcount, cols, testset), result_set, dists_set, epsilon);
}


void mexFunction(int nOutArray, mxArray *OutArray[], int nInArray, const mxArray *InArray[])
{
	/* Check the number of input arguments */
	if(nInArray != 4) {
		mexErrMsgTxt("Incorrect number of input arguments, expecting:\n"
		"dataset, testset, nearest_neighbors, epsilon");
		return;
	}

	/* Check the number of output arguments */
	if (nOutArray != 1 && nOutArray != 2) {
		mexErrMsgTxt("Incorrect number of outputs.");
		return;
	}

	const mxArray* datasetMat = InArray[0];
	const mxArray* testsetMat = InArray[1];

	if (!(mxIsSingle(datasetMat) && mxIsSingle(testsetMat))) {
		mexErrMsgTxt("Need single precision datasets for now...");
		return;
	}

	int dcount = mxGetN(datasetMat);
	int length = mxGetM(datasetMat);
	int tcount = mxGetN(testsetMat);

	if (mxGetM(testsetMat) != length) {
		mexErrMsgTxt("Dataset and testset features should have the same size.");
		return;
	}

	const mxArray* nnMat = InArray[2];

	if (mxGetM(nnMat)!=1 || mxGetN(nnMat)!=1 || !mxIsNumeric(nnMat)) {
		mexErrMsgTxt("Number of nearest neighbors should be a scalar.");
		return;
	}
	int nn = (int)(*mxGetPr(nnMat));

	float* dataset = (float*) mxGetData(datasetMat);
	float* testset = (float*) mxGetData(testsetMat);

	float epsilon = (float)mxGetScalar(InArray[3]);

    int* result = (int*)malloc(tcount*nn*sizeof(int));
    float* dists = (float*)malloc(tcount*nn*sizeof(float));

    /* do the search */
    flann_find_nearest_neighbors(dataset,dcount,length,testset, tcount, result, dists, nn, epsilon);

    /* Allocate memory for Output Matrix */
    OutArray[0] = mxCreateDoubleMatrix(nn, tcount, mxREAL);

    /* Get pointer to Output matrix and store result*/
    double* pOut = mxGetPr(OutArray[0]);
    for (int i=0;i<tcount*nn;++i) {
        pOut[i] = result[i]+1; // matlab uses 1-based indexing
    }
    free(result);

    if (nOutArray > 1) {
        /* Allocate memory for Output Matrix */
        OutArray[1] = mxCreateDoubleMatrix(nn, tcount, mxREAL);

        /* Get pointer to Output matrix and store result*/
        double* pDists = mxGetPr(OutArray[1]);
        for (int i=0;i<tcount*nn;++i) {
            pDists[i] = dists[i]; // matlab uses 1-based indexing
        }
    }
    free(dists);
}
