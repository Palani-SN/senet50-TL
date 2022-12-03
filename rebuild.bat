tar -xf test_set.zip
cd models
tar -xf train_set.zip
timeout /t 30
python build_model.py
timeout /t 20
python eval_model.py
cd ..
timeout /t 10
python test_model.py
@echo off
if not exist *.jpg (
echo Issue in execution, No JPG files generated in root folder
) else (
   echo Following JPG files have been created with results, Kindly check
   echo.
   dir /b *.jpg
   )