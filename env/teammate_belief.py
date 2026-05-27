"""Teammate-state belief module (v0.4+ ToM).

Will hold per-agent estimates of OTHER agents' positions and observed regions
between rendezvous events. Empty in v0.2; the obs schema's feat[5]
('prob_occupied') already carries the only teammate info used today.
"""
