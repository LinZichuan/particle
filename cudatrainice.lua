require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'

classes = {'1', '2', '3'}

local opt = lapp[[
    -n, --network   (default "")    reload pretrained network
    -b, --batchSize (default 128)    batch size
    -r, --learningRate  (default 0.0001)  learning rate, for SGD only
    -m, --momentum  (default 0) momentum, for SGD only
    -s, --save  (default '/home/lzc/particle/logs')
    -d, --data  (default 'gammas')
]]
if opt.network == '' then
    -- convnet
    -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
    model = nn.Sequential()
    model:add(nn.SpatialConvolutionMM(1, 6, 5, 5))   --1*6
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  --50*50

    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolutionMM(6, 16, 5, 5))   --6*16
    model:add(nn.Tanh()) 
    model:add(nn.SpatialMaxPooling(3, 3, 3, 3))

    -- stage 3 : standard 2-layer MLP:
    --model:add(nn.Dropout())
    model:add(nn.Reshape(16*14*14)) --1024
    model:add(nn.Linear(16*14*14, 200))
    model:add(nn.Tanh())
    model:add(nn.Linear(200, #classes))
    model:add(nn.LogSoftMax())
else
    print('<trainer> reloading previously trained network')
    model = torch.load(opt.network)
end

-- verbose
print(model)

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
trainLogger = optim.Logger(paths.concat(opt.save, 'traincudaice_2.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'testcudaice_2.log'))

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
    --trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
    local acc = confusion.totalValid * 100
    print (acc)
    os.execute('echo '..acc..' >> logs/train20150507.log' )
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
    --testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    local acc = confusion.totalValid*100
    os.execute('echo '..acc..' >> logs/test20150507.log' )
    confusion:zero()
end

--if opt.network=='' then 
    print('loading traindata and trainlabel...')
    local traindata = torch.load('./sample/gammas_2traindata_ice.t7')
    local trainlabel = torch.load('./sample/gammas_2trainlabel_ice.t7')
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
local testdata = torch.load('./sample/gammas_2testdata_ice.t7')
local testlabel = torch.load('./sample/gammas_2testlabel_ice.t7')
print (testdata:size())
print (testlabel:size())
print ('start testing..')
local testsize = testlabel:storage():size()
    print ('start training...')
    for i=1,100 do
        train(traindata, trainlabel, trainsize)
	    test(testdata, testlabel, testsize)
    end
    local trainsize = trainlabel:storage():size()
    print ('end training======================================================================================')
    print('saving current net...')
    --torch.save('network_cuda_ice_wider_300.t7', model)
    torch.save('network_cuda_ice_20160507.t7', model)
--end

local testda = torch.load('ice/ice.t7')
local siz = testda:storage():size() / 10000
local testla = torch.Tensor(siz):fill(3)
test(testda, testla, siz)


