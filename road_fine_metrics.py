"""Count narrow raster structures and reproduce the existing boundary metric."""
import numpy as np

from audit_road_density import boundary_band
from road_generalize_report import METRICS, contour, frame_counts, metrics


NAMES = METRICS + ("narrow_background_recall","narrow_road_recall","other_background_recall","other_road_recall")


def narrow_cells(frame):
    # A same-class run bounded by the opposite class on both sides in x or y.
    # At 0.5 m per cell a run of at most four cells spans at most 2 m.
    h,w=frame.shape
    padded=np.pad(frame,4,constant_values=255)
    def hit(dy,dx):
        dist=np.full((h,w),10,np.uint8)
        active=frame!=255
        for d in range(1,5):
            row,col=4+dy*d,4+dx*d
            neighbor=padded[row:row+h,col:col+w]
            opposite=(neighbor!=frame)&(neighbor!=255)
            dist[active & opposite]=d
            active=active & (neighbor==frame)
        return dist
    return ((hit(0,1)+hit(0,-1)<=5)|(hit(1,0)+hit(-1,0)<=5)) & (frame!=255)


def reference(y):
    return (boundary_band(y),[contour(row==1,row!=255) for row in y],np.stack([narrow_cells(row) for row in y]))


def fine_counts(pred,y,ref):
    band,gt,narrow=ref
    columns=[frame_counts(pred,y,band,gt)]
    for area in (narrow,~narrow):
        for label in (0,1):
            member=area & (y==label)
            columns.extend((((pred==label)&member).sum((1,2)),member.sum((1,2))))
    return np.column_stack(columns)


def scores(c):
    extra=c[...,12::2]/np.maximum(c[...,13::2],1)
    return np.concatenate((metrics(c[...,:12]),extra),axis=-1)
