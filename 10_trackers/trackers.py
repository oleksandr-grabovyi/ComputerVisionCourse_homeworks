import cv2


class TrackerTypes:
    CSRT = 0
    KCF = 1
    MIL = 2
    MOSSE = 3
    BOOSTING = 4


TRACKERS = {
    TrackerTypes.CSRT: cv2.TrackerCSRT,
    TrackerTypes.KCF: cv2.TrackerKCF,
    TrackerTypes.MIL: cv2.TrackerMIL,
    TrackerTypes.MOSSE: cv2.legacy.TrackerMOSSE,
    TrackerTypes.BOOSTING: cv2.legacy.TrackerBoosting,
}

def createTracker(trackerType):
    trackerCls = TRACKERS.get(trackerType)
    if trackerCls is not None:
        return trackerCls.create()
    return None
