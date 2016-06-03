require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'
require 'image'

nfeats = 1
width = 80
height = 80
ninputs = nfeats * width * height
classes = {'1', '2'}

--[[local opt = lapp[[
    -n, --network   (default "./net/hzhou_network_cuda_manual.t7")    reload pretrained network
    -b, --batchSize (default 32)    batch size
    -r, --learningRate  (default 0.0001)  learning rate, for SGD only
    -m, --momentum  (default 0) momentum, for SGD only
    -s, --save  (default '/home/lzc/particle/logs')
    -d, --data  (default 'gammas')
]]
local opt = {
	network="", 
	batchSize=32, 
	learningRate=0.01,--0.001, 
	momentum=0, 
	save="/home/lzc/particle/logs", 
	data="gammas"
}
if opt.network == '' then
    -- convnet
    model = nn.Sequential()

    model:add(nn.SpatialConvolutionMM(1, 16, 5, 5))   --1*6
	--model:add(nn.Dropout())
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  --38*38 --

    model:add(nn.SpatialConvolutionMM(16, 64, 5, 5))   --6*16
	--model:add(nn.Dropout(0.4))
    model:add(nn.Tanh()) 
    model:add(nn.SpatialMaxPooling(3, 3, 3, 3))  --11*11

    model:add(nn.Reshape(64*11*11)) --1024
    model:add(nn.Linear(64*11*11, 256))
	--model:add(nn.Dropout())
    model:add(nn.Tanh())
    model:add(nn.Linear(256, #classes))  

	model:add(nn.LogSoftMax())
	--[[model:add(nn.Reshape(80*80))
	model:add(nn.Dropout())
	model:add(nn.Linear(80*80, 1024))
	model:add(nn.Tanh())

	model:add(nn.Dropout())
	model:add(nn.Linear(1024, 512))
	model:add(nn.Tanh())

	model:add(nn.Dropout())
	model:add(nn.Linear(512, 256))
	model:add(nn.Tanh())

	model:add(nn.Linear(256, #classes)) ]]
else
    print('<trainer> reloading previously trained network')
    model = torch.load(opt.network)
end

-- verbose
--print(model)

-- loss function: negative log-likelihood
--
model:cuda()
criterion = nn.ClassNLLCriterion():cuda()

-- retrieve parameters and gradients
-- PARAMETERS MUST BE PUT ON CUDA!!!!!!!!!!!!!!!!!!! THIS OPERATION MUST BE PLACE AFTER MODEL:CUDA()
parameters,gradParameters = model:getParameters()

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
--trainLogger = optim.Logger(paths.concat(opt.save, 'traincuda_100_epoch_dropout.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'testcuda_100_epoch_dropout.log'))
--
--trainLogger = optim.Logger(paths.concat(opt.save, 'traincuda_raw.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'testcuda_raw.log'))

function train(dataset, label, size)
    epoch = epoch or 1
    --print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    times = 0
    for t = 1, size, opt.batchSize do
		xlua.progress(t, size)
        if t+opt.batchSize > size then
			xlua.progress(1, 1)
            break
        end
        --print ('times = ' .. times)
        --print ('t = ' .. t .. ', size = '..size..', minibatch = '..opt.batchSize)
        times = times + 1 
        -- create mini batch
        local inputs = torch.CudaTensor(opt.batchSize, 1, width, height)
        local targets = torch.CudaTensor(opt.batchSize)
        local k = 1
        for i = t, math.min(t+opt.batchSize-1, size) do
            -- load new sample
            local input = dataset[i]:clone()
            local target = label[i]
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        local feval = function(x)
            collectgarbage()
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end
            -- reset gradients
            gradParameters:zero()
            local f = 0
            for i = 1, opt.batchSize do
				local outputs = model:forward(inputs[i])
				local err = criterion:forward(outputs, targets[i])
        		f = f + err
				local df_do = criterion:backward(outputs, targets[i])
				model:backward(inputs[i], df_do:cuda()) 
				-- update confusion
	        	confusion:add(outputs:view(2), targets[i])
            end
            gradParameters:div(opt.batchSize)
            f = f / opt.batchSize
            return f, gradParameters
        end
        -- optimize on current mini-batch
        sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
        }
        optim.sgd(feval, parameters, sgdState)
        --optim.adam(feval, parameters)
        --xlua.progress(t, size)
		--[[local w1 = model:get(4).weight:float()
		local si = image.scale(w1, w1:size(2)*10, 'simple')
		if epoch == 1 then
			image.save('./changing/epoch'..epoch..'.t7', si)
			itorch.image(si)
		elseif epoch == 100 then 
			image.save('./changing/epoch'..epoch..'.t7', si)
			itorch.image(si)
		end	]]
    end
    print (confusion)
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    print (confusion.totalValid * 100)
    confusion:zero()
    epoch = epoch + 1
end

function test(testdata, testlabel, size)
    local inputs = torch.CudaTensor(size, 1, width, height)
    local targets = torch.CudaTensor(size)
    for i=1,size do
        inputs[i] = testdata[i]
        targets[i] = testlabel[i]
    end
    local outputs = model:forward(inputs)
    local f = criterion:forward(outputs, targets)
    for i = 1, size do
        confusion:add(outputs[i], targets[i])
        --print(outputs[i])
    end
    print(confusion)
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    confusion:zero()
end


--print('loading traindata and trainlabel...')
local traindata = torch.load('./sample/hzhou_traindata_manual.t7')
local trainlabel = torch.load('./sample/hzhou_trainlabel_manual.t7')
local trainsize = trainlabel:storage():size()
--print (traindata:size())
--print (trainlabel:size())
--[[local augmentdata = torch.load('./sample/no_overlap_testdata.t7')
local augmentlabel = torch.load('./sample/no_overlap_testlabel.t7')
local augsize = augmentlabel:storage():size()
print (augmentdata:size())
traindata = traindata:view(trainsize*10000):cat(augmentdata:view(augsize*10000)):view(trainsize+augsize, 100, 100)
print (augmentlabel:size())
trainlabel = trainlabel:cat(augmentlabel)]]
local testdata = torch.load('./sample/hzhou_testdata_manual.t7')
local testlabel = torch.load('./sample/hzhou_testlabel_manual.t7')
--print (testdata:size())
--print (testlabel:size())
--print ('start testing..')
local testsize = testlabel:storage():size()
--print ('start training...')
for i=1,100 do
	train(traindata, trainlabel, trainsize)
	if i % 5 == 0 then
		opt.learningRate = opt.learningRate / 5
	end
	test(testdata, testlabel, testsize)
end
local trainsize = trainlabel:storage():size()
print ('end training======================================================================================')


print('saving current net...')
--torch.save('./net/hzhou_network_cuda_manual_100_dropout.t7', model)
--torch.save('./net/hzhou_network_cuda_manual_0.9momentum.t7', model)
torch.save('./net/hzhou_network_cuda_manual_lr_decay.t7', model)


--torch.save('./filter/w1.t7', model:get(1).weight:float())
--torch.save('./filter/w4.t7', model:get(4).weight:float())


