# TOD
캡스톤 설계 27조

2013722049변주형

2013722059김유성

2013722067남건욱

2012722063손상렬



# 광학 및 열상 카메라를 이용한 객체 탐지 및 추적 시스템



## 설계 목표

![설계목표](https://user-images.githubusercontent.com/37025923/41344093-ce004fb2-6eef-11e8-83d7-769e28d842a4.PNG)

  구현하고자 하는 시스템은 IR/EO센서를 통해 같은 영상을 열 영상과, 일반 칼라영상으로 취득한 후 Object detection network를 통과시키기 전에 사람의 검출률을 향상시키기 위해 열 영상과 칼라 영상을 매칭 및 정합하는 Pre processing을 거친 후, Object detection network를 통해 사람을 검출하는 시스템이다.



## 사용하는 오픈소스SW

#### YOLO:RealTimeObjectDetection(<https://github.com/pjreddie/darknet>)

#### OpenCV: Open Source Computer Vision Library(https://github.com/opencv/opencv)

