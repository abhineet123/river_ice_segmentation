#pragma once

#include "Dataset.h"
#include "Allocator.h"
#include "ResultSet.h"



class PKDTree
{
private:

	enum {
		/**
		 * To improve efficiency, only SAMPLE_MEAN random values are used to
		 * compute the mean and variance at each level when building a tree.
		 * A value of 100 seems to perform as well as using all values.
		 */
		SAMPLE_MEAN = 100,
		/**
		 * Top random dimensions to consider
		 *
		 * When creating random trees, the dimension on which to subdivide is
		 * selected at random from among the top RAND_DIM dimensions with the
		 * highest variance.  A value of 5 works well.
		 */
		RAND_DIM=5
	};



	class Node {
	public:


		/**
		 * Index of the vector feature used for subdivision.
		 * If this is a leaf node (both children are NULL) then
		 * this holds vector index for this leaf.
		 */
		int divfeat;
		/**
		 * The value used for subdivision.
		 */
		float divval;
		/**
		 * The child nodes.
		 */
		Node *child1, *child2;
	};

	/**
	 * The dataset used by this index
	 */
	Dataset<float>& m_dataset;

	/**
	 * The root node
	 */
	Node* m_root;

	/**
	 *  Array of indices to vectors in the dataset.  When doing lookup,
	 *  this is used instead to mark checkID.
	 */
	int* m_vind;


	/**
	 * Pooled memory allocator.
	 *
	 * Using a pooled memory allocator is more efficient
	 * than allocating memory directly when there is a large
	 * number small of memory allocations.
	 */
	PooledAllocator m_pool;

    int m_size;
    int m_dims;

	float* m_mean;
    float* m_var;

public:
	PKDTree(Dataset<float>& inputData) : m_dataset(inputData), m_root(NULL)
	{
		m_size = m_dataset.rows;
        m_dims = m_dataset.cols;

		// Create a permutable array of indices to the input vectors.
		m_vind = new int[m_size];
		for (int i = 0; i < m_size; i++) {
			m_vind[i] = i;
		}

		m_mean = new float[m_dims];
        m_var = new float[m_dims];

		build();
	}
	~PKDTree(void)
	{
		delete [] m_vind;
		delete [] m_mean;
		delete [] m_var;
	}

private:

	/*************** TREE CONSTRUCTION ********************/
	void build()
	{
		divideTree(&m_root, 0, m_size - 1);
	}

	
	/**
	 * Create a tree node that subdivides the list of vecs from vind[first]
	 * to vind[last].  The routine is called recursively on each sublist.
	 * Place a pointer to this new tree node in the location pTree.
	 *
	 * Params: pTree = the new node to create
	 * 			first = index of the first vector
	 * 			last = index of the last vector
	 */
	void divideTree(Node** pTree, int first, int last)
	{
		Node* node;

		node = m_pool.allocate<Node>(); // allocate memory
		*pTree = node;

		/* If only one exemplar remains, then make this a leaf node. */
		if (first == last) {
			node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
			node->divfeat = m_vind[first];    /* Store index of this vec. */
		} else {
			chooseDivision(node, first, last);
			subdivide(node, first, last);
		}
	}

		
	/**
	 * Choose which feature to use in order to subdivide this set of vectors.
	 * Make a random choice among those with the highest variance, and use
	 * its variance as the threshold value.
	 */
	void chooseDivision(Node* node, int first, int last)
	{
        memset(m_mean,0,m_dims*sizeof(float));
        memset(m_var,0,m_dims*sizeof(float));

		/* Compute mean values.  Only the first SAMPLE_MEAN values need to be
			sampled to get a good estimate.
		*/
		int end = min(first + SAMPLE_MEAN, last);
		int count = end - first + 1;
		for (int j = first; j <= end; ++j) {
			float* v = m_dataset[m_vind[j]];
            for (int k=0; k<m_dims; ++k) {
                m_mean[k] += v[k];
            }
		}
        for (int k=0; k<m_dims; ++k) {
            m_mean[k] /= count;
        }

		/* Compute variances (no need to divide by count). */
		for (int j = first; j <= end; ++j) {
			float* v = m_dataset[m_vind[j]];
            for (int k=0; k<m_dims; ++k) {
                float dist = v[k] - m_mean[k];
                m_var[k] += dist * dist;
            }
		}
		/* Select one of the highest variance indices at random. */
		node->divfeat = selectDivision(m_var);
		node->divval = m_mean[node->divfeat];
	}


	/**
	 * Select the top RAND_DIM largest values from v and return the index of
	 * one of these selected at random.
	 */
	int selectDivision(float* v)
	{
		int num = 0;
		int topind[RAND_DIM];

		/* Create a list of the indices of the top RAND_DIM values. */
		for (int i = 0; i < m_dims; ++i) {
			if (num < RAND_DIM  ||  v[i] > v[topind[num-1]]) {
				/* Put this element at end of topind. */
				if (num < RAND_DIM) {
					topind[num++] = i;            /* Add to list. */
				}
				else {
					topind[num-1] = i;         /* Replace last element. */
				}
				/* Bubble end value down to right location by repeated swapping. */
				int j = num - 1;
				while (j > 0  &&  v[topind[j]] > v[topind[j-1]]) {
					swap(topind[j], topind[j-1]);
					--j;
				}
			}
		}
		/* Select a random integer in range [0,num-1], and return that index. */
// 		int rand = cast(int) (drand48() * num);
		int rnd = rand_int(num);
		assert(rnd >=0 && rnd < num);
		return topind[rnd];
	}

		/**
	 *  Subdivide the list of exemplars using the feature and division
	 *  value given in this node.  Call divideTree recursively on each list.
	*/
	void subdivide(Node* node, int first, int last)
	{
		/* Move vector indices for left subtree to front of list. */
		int i = first;
		int j = last;
		while (i <= j) {
			int ind = m_vind[i];
			float val = m_dataset[ind][node->divfeat];
			if (val < node->divval) {
				++i;
			} else {
				/* Move to end of list by swapping vind i and j. */
				swap(m_vind[i], m_vind[j]);
				--j;
			}
		}
		/* If either list is empty, it means we have hit the unlikely case
			in which all remaining features are identical. Split in the middle
            to maintain a balanced tree.
		*/
		if ( (i == first) || (i == last+1)) {
            i = (first+last+1)/2;
		}

		divideTree(& node->child1, first, i - 1);
		divideTree(& node->child2, i, last);
	}

	/*************** END: TREE CONSTRUCTION ********************/


	/*************** NEIGHBORE SEARCH *******************/


public:
	/**
	 * Performs an nearest neighbor search.
	 */
	void getNeighbors(ResultSet& result, float* vec, float epsilon)
	{

		float* min_vect = new float[m_dims];
		memcpy(min_vect, vec, m_dims*sizeof(float));
		epsilon = (1+epsilon)*(1+epsilon);
		searchLevel(result, vec, m_root, 0.0, min_vect, epsilon);
		assert(result.full());
		delete [] min_vect;
	}

private:
	/**
	 * Performs a search in the tree starting from a node.
	 */
	void searchLevel(ResultSet& result, float* vec, Node* node, float mindistsq, float* min_vect, float epsilon)
	{
		if (epsilon*mindistsq>result.worstDist()) {
			return;
		}

		float val, diff;
		Node* bestChild, *otherChild;

		/* If this is a leaf node, then do check and return. */
		if (node->child1 == NULL  &&  node->child2 == NULL) 
		{
			result.addPoint(m_dataset[node->divfeat],node->divfeat);
		}
		else
		{
			/* Which child branch should be taken first? */
			val = vec[node->divfeat];
			diff = val - node->divval;
			bestChild = (diff < 0) ? node->child1 : node->child2;
			otherChild = (diff < 0) ? node->child2 : node->child1;


			/* Call recursively to search next level down. */
			searchLevel(result, vec, bestChild, mindistsq, min_vect, epsilon);

			double new_distsq = mindistsq;
			new_distsq -= flann_dist(&val, &val+1, &min_vect[node->divfeat], 0);
			float tmp_min_val = min_vect[node->divfeat];
			min_vect[node->divfeat] = node->divval;
			new_distsq += flann_dist(&val, &val+1, &min_vect[node->divfeat], 0);
//			double new_distsq = flann_dist(&val, &val+1, &node->divval, mindistsq);

			searchLevel(result, vec, otherChild, (float)new_distsq, min_vect, epsilon);

			min_vect[node->divfeat] = tmp_min_val;
		}
	}

};
