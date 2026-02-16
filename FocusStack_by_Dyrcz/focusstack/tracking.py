from dataclasses import dataclass, field
from typing import List, Dict
from .refs import RefObj
from .segmentation import SegMask

@dataclass
class TrackDet:
    t: int
    mask01: np.ndarray
    bbox_xyxy: tuple
    area: int
    score: float

@dataclass
class Track:
    track_id: int
    dets: List[TrackDet] = field(default_factory=list)

def tracks_from_assignments(refs: List[RefObj], assigns_per_frame: List[Dict[int, SegMask]]) -> List[Track]:
    tracks: Dict[int, Track] = {r.ref_id: Track(track_id=r.ref_id) for r in refs}
    for t, amap in enumerate(assigns_per_frame):
        for ref_id, m in amap.items():
            tracks[ref_id].dets.append(
                TrackDet(
                    t=t,
                    mask01=m.mask01.astype(np.uint8),
                    bbox_xyxy=m.bbox_xyxy,
                    area=m.area,
                    score=m.score,
                )
            )
    return [tr for tr in tracks.values() if tr.dets]
