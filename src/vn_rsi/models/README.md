# Trust Level: AGENT-MUTABLE WEIGHTS (TRAINED STRICTLY ON SEARCH-ZONE VERIFIED TRAJECTORIES)
# Models here (`DiscreteASTDiffusion`, `HRMReasoner`, `ModularMicroRSIBackend`) are strictly prohibited
# from accessing `validation` or `sealed_test` raw bars; they receive search-zone trajectories and
# scalar validation fitness ONLY through the human-owned SealedEvaluator interface.
