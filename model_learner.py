import xlearn as xl

ffm_model = xl.create_ffm()
ffm_model.setTrain('./traindata.libsvm')
ffm_model.setValidate('./testdata.libsvm')
param = {'task':'binary', 'lr':0.2,
         'lambda':0.002, 'metric':'acc'}

ffm_model.fit(param, './model.out')


# Prediction task
ffm_model.setTest("./testdata.libsvm")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./model.out", "./output.txt")