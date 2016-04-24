# PolyU-GuidedStudy-CNN

This is a course project for PolyU COMP 6815.

#### System
I only test this code on Win8.1 + VS2013.

#### Dependencies
- OpenCV 3.0 (or later)
- ArrayFire

##### Warning: This project is not well-optimized since it is only for understanding some detail operations in CNNs. I highly suggest that everyone should not use this code to train a large-scale network on large-scale dataset. Please use other excellent pacakges, such as Caffe, MxNet, MatConvnet, Cudnn etc.

#### If your still want to use it, please read:

1. Install OpenCV3.0 and ArrayFie.
2. Create a new Project in VS2013, and add all "CNN/*.h" and "CNN/.cpp" in "header" and "source" fold in VS2013.
3. Make sure all "include" and "lib" in the right place. (You should include ArrayFire and OpenCV).
4. Add "opencv_world300.lib" (or "opencv_world300d.lib") and "afcpu.lib" into "Additional Dependencies".
5. Add "test/testMNIST" or "test/testCIFAR10" into "source" fold and run the project.

Now, this code can only run on CPU, so it is a little slower.

#### PS: If you have any questions in CNNs, please feel free to discuss with me -- lingxiao.yang717@gmail.com
#### PPS: I am also a freshman (^_^).
