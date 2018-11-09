PRIM-PE
=======

Python implementation of the good enough region algorithm from [Shokri2016]_.
Considering :math:`f(X) = Y`, it samples uniformly the input parameter space so that the output exceed a threashold.
The algorithm is as follows:

1. Sample uniformly the input parameter space: :math:`X`,
2. Simulate the sample: :math:`f(X) = Y`,
3. Categorize the sample into successful and failed one,
4. Sort the population based on threashold value and density,
5. Select successful and failed parents to sample from,
6. Sample new points around parents,
7. Iterate.
   
This method has been showned to be more efficient than DREAM.

.. [Shokri2016] Ashkan Shokri et al. Application of the patient rule induction method to detect hydrologic model behavioural parameters and quantify uncertainty. Hydrological Processes. 2016. DOI: 10.1002/hyp.11464
