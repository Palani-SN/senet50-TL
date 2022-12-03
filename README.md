# senet5 - Transfer Learning

- Example code to build a customized face prediction model based on vgg16-senet50 pretrained model to identify faces from trained dataset, output of the code looks as shown below

	![](https://github.com/Palani-SN/senet50-TL/blob/main/images/result1.jpg?raw=true) 

## Steps to execute

- open command prompt and run **pip install -r requirements.txt** to install the packages required

- Try running **run_demo.bat** for testing the predictions on a test_set available, 
you can be able to see the results of the predictions in the root folder, once the execution is done.

```terminal

Microsoft Windows [Version 10.0.19044.2251]
(c) Microsoft Corporation. All rights reserved.

F:\GitRepos\senet50-TL>run_demo.bat

F:\GitRepos\senet50-TL>tar -xf test_set.zip

F:\GitRepos\senet50-TL>cd models

F:\GitRepos\senet50-TL\models>tar -xf train_set.zip

F:\GitRepos\senet50-TL\models>tar -xf models.zip

F:\GitRepos\senet50-TL\models>timeout /t 30

Waiting for  0 seconds, press a key to continue ...

F:\GitRepos\senet50-TL\models>python eval_model.py
2022-12-03 22:08:17.476674: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-12-03 22:08:17.476872: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2022-12-03 22:08:44.721672: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-12-03 22:08:44.723804: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublas64_11.dll'; dlerror: cublas64_11.dll not found
2022-12-03 22:08:44.725591: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublasLt64_11.dll'; dlerror: cublasLt64_11.dll not found
2022-12-03 22:08:44.727301: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2022-12-03 22:08:44.729028: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2022-12-03 22:08:44.730747: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusolver64_11.dll'; dlerror: cusolver64_11.dll not found
2022-12-03 22:08:44.732798: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusparse64_11.dll'; dlerror: cusparse64_11.dll not found
2022-12-03 22:08:44.734485: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudnn64_8.dll'; dlerror: cudnn64_8.dll not found
2022-12-03 22:08:44.734673: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2022-12-03 22:08:44.865124: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
--> F:\GitRepos\senet50-TL\models\eval_set\chandler_medieval.png (1, 224, 224, 3)
actual : chandler
expected : chandler
--> F:\GitRepos\senet50-TL\models\eval_set\chandler_old.png (1, 224, 224, 3)
actual : chandler
expected : chandler
--> F:\GitRepos\senet50-TL\models\eval_set\chandler_young.png (1, 224, 224, 3)
actual : chandler
expected : chandler
--> F:\GitRepos\senet50-TL\models\eval_set\joey_medieval.png (1, 224, 224, 3)
actual : joey
expected : joey
--> F:\GitRepos\senet50-TL\models\eval_set\joey_old.png (1, 224, 224, 3)
actual : joey
expected : joey
--> F:\GitRepos\senet50-TL\models\eval_set\joey_young.png (1, 224, 224, 3)
actual : joey
expected : joey
--> F:\GitRepos\senet50-TL\models\eval_set\monika_medieval.png (1, 224, 224, 3)
actual : monika
expected : monika
--> F:\GitRepos\senet50-TL\models\eval_set\monika_old.png (1, 224, 224, 3)
actual : monika
expected : monika
--> F:\GitRepos\senet50-TL\models\eval_set\monika_young.png (1, 224, 224, 3)
actual : monika
expected : monika
--> F:\GitRepos\senet50-TL\models\eval_set\phoebe_medieval.png (1, 224, 224, 3)
actual : phoebe
expected : phoebe
--> F:\GitRepos\senet50-TL\models\eval_set\phoebe_old.png (1, 224, 224, 3)
actual : phoebe
expected : phoebe
--> F:\GitRepos\senet50-TL\models\eval_set\phoebe_young.png (1, 224, 224, 3)
actual : phoebe
expected : phoebe
--> F:\GitRepos\senet50-TL\models\eval_set\rachel_medieval.png (1, 224, 224, 3)
actual : rachel
expected : rachel
--> F:\GitRepos\senet50-TL\models\eval_set\rachel_old.png (1, 224, 224, 3)
actual : rachel
expected : rachel
--> F:\GitRepos\senet50-TL\models\eval_set\rachel_young.png (1, 224, 224, 3)
actual : rachel
expected : rachel
--> F:\GitRepos\senet50-TL\models\eval_set\ross_medieval.png (1, 224, 224, 3)
actual : ross
expected : ross
--> F:\GitRepos\senet50-TL\models\eval_set\ross_old.png (1, 224, 224, 3)
actual : ross
expected : ross
--> F:\GitRepos\senet50-TL\models\eval_set\ross_young.png (1, 224, 224, 3)
actual : ross
expected : ross
[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
1/1 [==============================] - 5s 5s/step - loss: 0.0178 - accuracy: 1.0000
loss :  0.01779872179031372 acc :  1.0

F:\GitRepos\senet50-TL\models>cd ..

F:\GitRepos\senet50-TL>timeout /t 10

Waiting for  0 seconds, press a key to continue ...

F:\GitRepos\senet50-TL>python test_model.py
2022-12-03 22:09:44.731193: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-12-03 22:09:44.731417: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2022-12-03 22:10:05.572566: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2022-12-03 22:10:05.574565: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublas64_11.dll'; dlerror: cublas64_11.dll not found
2022-12-03 22:10:05.576258: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cublasLt64_11.dll'; dlerror: cublasLt64_11.dll not found
2022-12-03 22:10:05.578351: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2022-12-03 22:10:05.580127: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2022-12-03 22:10:05.581901: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusolver64_11.dll'; dlerror: cusolver64_11.dll not found
2022-12-03 22:10:05.583716: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cusparse64_11.dll'; dlerror: cusparse64_11.dll not found
2022-12-03 22:10:05.586049: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'cudnn64_8.dll'; dlerror: cudnn64_8.dll not found
2022-12-03 22:10:05.586283: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1850] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2022-12-03 22:10:05.587499: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Following JPG files have been created with results, Kindly check

output-chandler-recent.jpg
output-friends-medieval.jpg
output-friends-new.jpg
output-friends-old.jpg
output-joey-recent.jpg
output-monika-recent.jpg
output-phoebe-recent.jpg
output-rachel-recent.jpg
output-ross-recent.jpg

F:\GitRepos\senet50-TL>

```

- To rebuild the model, Try running rebuild.bat the execution might be slow based on hardware, and the resultant model might differ in terms of performance.

	- example usage : **rebuild.bat** (for results - see build_model.log for reference)
	
## Additional results

- Group Images

	![](https://github.com/Palani-SN/senet50-TL/blob/main/images/result2.jpg?raw=true)
	
- Identification of persons in recent Images

	![](https://github.com/Palani-SN/senet50-TL/blob/main/images/aged1.jpg?raw=true)
	
	![](https://github.com/Palani-SN/senet50-TL/blob/main/images/aged2.jpg?raw=true)
