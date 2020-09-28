# Weighted Boxes Fusion: ensembling boxes for object detection models 

> In this work, we introduce a novel Weighted Box Fusion (WBF) ensembling algorithm that boosts the performance by ensembling predictions from different object detection models. 

### Alogorithm

**Input:** list of boxes A = [[x1, y1, x2, y2, score], ..., [x1, y1, x2, y2, score]], the shape is (n, 5) and sort the list     			 with score
             iou_threshold
             score_threshold
1: initialize list of L <-- ∅
2: initialize list of F <-- ∅
3: initialize list of R <-- ∅
4: while length of A is > 0:
5:	compute iou I
6:	if I > iou_threshold:
7:		add matched boxes M to the L at the pos of F
8:		delete the matched boxes from A
9:	if there are no matched boxes:
10:		add the start box of A to the end of L and F
11:  when A is ∅:
12:		compute fusion boxes and scores
13:		add fusion boxes and scores to R
14: return R

<img src="/illustration/1.png" align="center" width="400">

<img src="/illustration/2.png" align="center" width="300">