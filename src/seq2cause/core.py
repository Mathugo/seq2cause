class InstanceLevelCausalDiscovery:
    """ "
    This class implements instance-level causal discovery algorithm.
    When given a batch of discrete sequence of events, e.g., "A B C D",
    it identifies the instance time and summary causal graph per sequence.

    Attributes:
        tfx (any): The autoregressive model used to compute next-token probabilities.
    """
