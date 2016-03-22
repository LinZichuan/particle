require 'torch'
require 'nn'
require 'optim'

nfeats = 1
width = 100
height = 100
ninputs = nfeats * width * height
classes = {'1', '2'}

local opt = lapp[[
    -n, --network   (default "aug_network.t7")    reload pretrained network
    -b, --batchSize (default 64)    batch size
    -r, --learningRate  (default 0.001)  learning rate, for SGD only
    -m, --momentum  (default 0) momentum, for SGD only
]]
if opt.network == '' then
    -- convnet
    -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
    model = nn.Sequential()
    model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
    model:add(nn.Tanh())
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  --50*50

    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
    model:add(nn.Tanh()) 
    model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- stage 3 : standard 2-layer MLP:
    model:add(nn.Reshape(64*14*14))
    model:add(nn.Linear(64*14*14, 1000))
    model:add(nn.Tanh())
    model:add(nn.Linear(1000, 200))
    model:add(nn.Tanh())
    model:add(nn.Linear(200, #classes))
else
    print('<trainer> reloading previously trained network')
    model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print(model)

-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

function train(dataset, label, size)
    epoch = epoch or 1
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    times = 1
    for t = 1, size, opt.batchSize do
        if t+opt.batchSize > size then
            return
        end
        print ('times = ' .. times)
        print ('t = ' .. t .. ', size = '..size..', minibatch = '..opt.batchSize)
        times = times + 1 
        -- create mini batch
        local inputs = torch.Tensor(opt.batchSize, 1, 100, 100)
        local targets = torch.Tensor(opt.batchSize)
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
            -- evaluate function for complete mini batch
            local outputs = model:forward(inputs)
            --print(outputs)
            local f = criterion:forward(outputs, targets)
            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do) 
            -- update confusion
            for i = 1,math.min(t+opt.batchSize-1,size)-t do
                confusion:add(outputs[i], targets[i])
            end
            return f, gradParameters
        end
        -- optimize on current mini-batch
        sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
        }
        optim.sgd(feval, parameters, sgdState)
        
        xlua.progress(t, size)
        
        print(confusion)
        confusion:zero()

    end
    epoch = epoch + 1
end

function test(testdata, testlabel, size)
    local inputs = torch.Tensor(size, 1, 100, 100)
    local targets = torch.Tensor(size)
    for i=1,size do
        inputs[i] = testdata[i]
        targets[i] = testlabel[i]
    end
    local outputs = model:forward(inputs)
    local f = criterion:forward(outputs, targets)
    for i = 1, size do
        confusion:add(outputs[i], targets[i])
    end
    print(confusion)
    confusion:zero()
end

print('loading traindata and trainlabel...')
local traindata = torch.load('./sample/2traindata.t7')
local trainlabel = torch.load('./sample/2trainlabel.t7')
local trainsize = trainlabel:storage():size()
print (traindata:size())
print (trainlabel:size())
local augmentdata = torch.load('./sample/augment_traindata.t7')
local augmentlabel = torch.load('./sample/augment_trainlabel.t7')
local augsize = augmentlabel:storage():size()
print (augmentdata:size())
traindata = traindata:view(trainsize*10000):cat(augmentdata:view(augsize*10000)):view(trainsize+augsize, 100, 100)
print (augmentlabel:size())
trainlabel = trainlabel:cat(augmentlabel)
collectgarbage()

print ('start training...')
local trainsize = trainlabel:storage():size()
train(traindata, trainlabel, trainsize)
print ('end training======================================================================================')

local testdata = torch.load('./sample/2testdata.t7')
local testlabel = torch.load('./sample/2testlabel.t7')
print ('start testing..')
local testsize = testlabel:storage():size()
test(testdata, testlabel, testsize)

print('saving current net...')
torch.save('aug_network.t7', model)



