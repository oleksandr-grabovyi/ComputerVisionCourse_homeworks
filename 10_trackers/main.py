import cv2
import numpy as np
from trackers import TrackerTypes, createTracker

"""
if there are any problems with import, probably wrong opencv package installed
pip uninstall opencv-python
pip install opencv-contrib-python
"""
trackers = [
    createTracker(TrackerTypes.CSRT),
    #createTracker(TrackerTypes.BOOSTING),
    createTracker(TrackerTypes.MIL),
]

video = cv2.VideoCapture('video.mp4')
_, frame = video.read()

# initial_bbox = cv2.selectROI(frame, False)  # allows to set up bbox manually in the opened window
initial_bbox = (152, 332, 50, 76)

# I want to glue several trackers and videos in 1 for providing of the some comparison
for tracker in trackers:
    tracker.init(frame, initial_bbox)

_, video_width, _ = frame.shape

while True:
    ret, frame = video.read()
    if not ret:
        break

    # glue frames in 1
    combined_frame = np.hstack([frame for i in range(len(trackers))])
    for index, tracker in enumerate(trackers):
        # I use several trackers for comparison
        ret, bbox = tracker.update(frame)
        if ret:
            left_top = (int(bbox[0]) + index * video_width, int(bbox[1]))
            right_bottom = (int(bbox[0] + bbox[2]) + index * video_width, int(bbox[1] + bbox[3]))
            cv2.rectangle(combined_frame, left_top, right_bottom, (255, 0, 0), 2, 1)
            cv2.putText(combined_frame,
                        str(tracker).split()[1],  # name of the tracker class
                        (10 + index * video_width, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Tracking", combined_frame)
    # Generally we need here some delay,
    # but trackers execution is bigger that normal FPS therefore there is no reason to set extra delay
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
