require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'

nfeats = 1
width = 100
height = 100
ninputs = nfeats * width * height
classes = {'1', '2'}

local opt = lapp[[
    -n, --network   (default "")    reload pretrained network
    -b, --batchSize (default 32)    batch size
    -r, --learningRate  (default 0.01)  learning rate, for SGD only
    -m, --momentum  (default 0.9) momentum, for SGD only
    -s, --save  (default '/home/lzc/particle/logs')
    -d, --data  (default 'gammas')
]]
if opt.network == '' then
    -- convnet
    -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
    model = nn.Sequential()
    model:add(nn.SpatialConvolutionMM(1, 4, 5, 5))   --1*6
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  --50*50

    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolutionMM(4, 9, 5, 5))   --6*16
	model:add(nn.Dropout(0.4))
    model:add(nn.Tanh()) 
    model:add(nn.SpatialMaxPooling(3, 3, 3, 3))

    -- stage 3 : standard 2-layer MLP:
    --model:add(nn.Dropout())
    model:add(nn.Reshape(9*14*14)) --1024
    model:add(nn.Linear(9*14*14, 200))
	model:add(nn.Dropout(0.4))
    model:add(nn.Tanh())
    model:add(nn.Linear(200, #classes))
else
    print('<trainer> reloading previously trained network')
    model = torch.load(opt.network)
end

-- verbose
print(model)

-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
model:cuda()
criterion = nn.ClassNLLCriterion():cuda()

-- retrieve parameters and gradients
-- PARAMETERS MUST BE PUT ON CUDA!!!!!!!!!!!!!!!!!!! THIS OPERATION MUST BE PLACE AFTER MODEL:CUDA()
parameters,gradParameters = model:getParameters()

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'traincuda.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'testcuda.log'))

function train(dataset, label, size)
    epoch = epoch or 1
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    times = 1
    for t = 1, size, opt.batchSize do
        if t+opt.batchSize > size then
            break
        end
        --print ('times = ' .. times)
        --print ('t = ' .. t .. ', size = '..size..', minibatch = '..opt.batchSize)
        times = times + 1 
        -- create mini batch
        local inputs = torch.CudaTensor(opt.batchSize, 1, 100, 100)
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
	        confusion:add(outputs, targets[i])
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
    end
    print (confusion)
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    print (confusion.totalValid * 100)
    confusion:zero()
    epoch = epoch + 1
end

function test(testdata, testlabel, size)
    local inputs = torch.CudaTensor(size, 1, 100, 100)
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

if opt.network=='' then 
    print('loading traindata and trainlabel...')
    local traindata = torch.load('./sample/2traindata.t7')
    local trainlabel = torch.load('./sample/2trainlabel.t7')
    local trainsize = trainlabel:storage():size()
    print (traindata:size())
    print (trainlabel:size())
    --[[local augmentdata = torch.load('./sample/no_overlap_testdata.t7')
    local augmentlabel = torch.load('./sample/no_overlap_testlabel.t7')
    local augsize = augmentlabel:storage():size()
    print (augmentdata:size())
    traindata = traindata:view(trainsize*10000):cat(augmentdata:view(augsize*10000)):view(trainsize+augsize, 100, 100)
    print (augmentlabel:size())
    trainlabel = trainlabel:cat(augmentlabel)]]
local testdata = torch.load('./sample/2testdata.t7')
local testlabel = torch.load('./sample/2testlabel.t7')
print (testdata:size())
print (testlabel:size())
print ('start testing..')
local testsize = testlabel:storage():size()
    print ('start training...')
    for i=1,100 do
		if i % 5 == 0 then
			opt.learningRate = opt.learningRate / 5
		end
        train(traindata, trainlabel, trainsize)
		test(testdata, testlabel, testsize)
    end
    local trainsize = trainlabel:storage():size()
    print ('end training======================================================================================')
end


print('saving current net...')
torch.save('network_cuda_20160606.t7', model)



