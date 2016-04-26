local a = torch.load('trainice.t7')
local b = torch.load('testice.t7')

local traindata = torch.load('../sample/icetraindata.t7')
local testdata = torch.load('../sample/icetestdata.t7')
local trainlabel = torch.load('../sample/icetrainlabel.t7')
local testlabel = torch.load('../sample/icetestlabel.t7')

local icetrainsize = a:storage():size() / 10000
local icetestsize = b:storage():size() / 10000
local origintrainsize = traindata:storage():size() / 10000
local origintestsize = testdata:storage():size() / 10000

traindata = traindata:view(traindata:storage():size()):cat(a:view(a:storage():size()))
testdata = testdata:view(testdata:storage():size()):cat(b:view(b:storage():size()))
trainlabel = trainlabel:cat(torch.Tensor(icetrainsize):fill(3))
testlabel = testlabel:cat(torch.Tensor(icetestsize):fill(3))

local trainsize = icetrainsize + origintrainsize
local testsize = icetestsize + origintestsize
traindata = traindata:view(trainsize, 100, 100)
testdata = testdata:view(testsize, 100, 100)

local shuffletraindata = torch.Tensor(trainsize, 100, 100)
local shuffletestdata = torch.Tensor(testsize, 100, 100)
local shuffletrainlabel = torch.Tensor(trainsize)
local shuffletestlabel = torch.Tensor(testsize)

local shuffletrain = torch.randperm(trainsize)
local shuffletest = torch.randperm(testsize)

for i = 1, trainsize do
	local idx = shuffletrain[i]
	shuffletraindata[i] = traindata[idx]
	shuffletrainlabel[i] = trainlabel[idx]
end
for i = 1, testsize do
	local idx = shuffletest[i]
	shuffletestdata[i] = testdata[idx]
	shuffletestlabel[i] = testlabel[idx]
end


torch.save('../sample/2traindata_ice.t7', shuffletraindata)
torch.save('../sample/2testdata_ice.t7', shuffletestdata)
torch.save('../sample/2trainlabel_ice.t7', shuffletrainlabel)
torch.save('../sample/2testlabel_ice.t7', shuffletestlabel)
