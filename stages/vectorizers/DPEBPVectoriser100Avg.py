from stages.vectorizers.components.DPEBPVectoriser import DPEBPVectoriser


class DPEBPVectoriser100Avg(DPEBPVectoriser):
    def __init__(self):
        super().__init__(dim=100, agg_metohd='avg')
