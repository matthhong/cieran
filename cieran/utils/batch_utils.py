"""
MIT License

Copyright (c) 2021 Stanford Intelligent and Interactive Autonomous Systems Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Utility functions for active batch generation.
"""

from typing import List
import numpy as np
import scipy.spatial.distance as ssd

from cieran.learning import Query, WeakComparisonQuery


def default_query_distance(queries: List[Query], **kwargs) -> np.array:
    """Given a set of m queries, returns an m-by-m matrix, each entry representing the distance between the corresponding queries.
    
    Args:
        queries (List[Query]): list of m queries for which the distances will be computed
        **kwargs: The hyperparameters.

            - `metric` (str): The distance metric can be specified with this argument. Defaults to 'euclidean'. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html for the set of available metrics.
                  
    Returns:
        numpy.array: an m-by-m numpy array that consists of the pairwise distances between the queries.
        
    Raises:
        AssertionError: if the query is not a compatible type. Cieran is only compatible with weak comparison queries.
    """
    kwargs.setdefault('metric', 'euclidean')
    compatible_types = [isinstance(query, WeakComparisonQuery) for query in queries]
    assert np.all(compatible_types), 'Default query distance, which you are using for batch selection, does not support the given query types. Consider using a custom distance function. See utils/batch_utils.py.'
    assert np.all([query.K == 2 for query in queries]), 'Default query distance, which you are using for batch selection, does not support large slates, use K = 2. Or consider using a custom distance function. See utils/batch_utils.py.'

    features_diff = [query.slate.features_matrix[0] - query.slate.features_matrix[1] for query in queries]
    return ssd.squareform(ssd.pdist(features_diff, kwargs['metric']))
