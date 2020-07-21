import xlearn as xl

ffm_model = xl.create_ffm()
ffm_model.setTrain('./train_data.libsvm')
ffm_model.setValidate('./validation_data.libsvm')
param = {'task':'binary', 'lr':0.2,
         'lambda':0.002, 'metric':'acc', 'k': 1000}

ffm_model.fit(param, './model.out')


# Prediction task
ffm_model.setTest("./test_data.libsvm")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./model.out", "./output.txt")