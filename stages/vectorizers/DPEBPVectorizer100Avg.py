from stages.vectorizers.components.DPEBPVectorizer import DPEBPVectorizer


class DPEBPVectorizer100Avg(DPEBPVectorizer):
    def __init__(self):
        super().__init__(dim=100, agg_metohd='avg')
