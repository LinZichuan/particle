a = torch.load('ice0003.t7')
b = torch.load('ice0060.t7')

res = a:view(a:storage():size()):cat(b:view(b:storage():size()))
size = res:storage():size() / 10000
print (size)
torch.save('ice.t7', res:view(size, 100, 100))
res = res:view(size, 100, 100)


trainnum = 6327 + 703

num = 1

local totaldata = torch.Tensor(trainnum, 100, 100)
function flip(ice)
	local res = torch.FloatTensor(6, 100, 100)
	for i = 1, 100 do
		for j = 1, 100 do
			res[1][j][101-i] = ice[i][j]
			res[2][101-i][101-j] = ice[i][j]
			res[3][101-j][i] = ice[i][j]
			res[4][i][101-j] = ice[i][j]
			res[5][101-i][j] = ice[i][j]
		end
	end
	res[6] = ice
	return res
end

while num < trainnum do
	local index = (num-1)%size + 1
	local ice = torch.Tensor(res[index]:size()):copy(res[index])
	--[[for i = 1, 100 do
		for j = 1, 100 do
			local r = torch.uniform()
			if r < 0.2 then
				ice[{i,j}] = 0   --dropout
			end
		end
	end]]
	local e = math.min(trainnum, num+6)
	local f = flip(ice)
	for i = num+1, e do
		totaldata[i] = f[i-num]
	end
	num = num + 6
	xlua.progress(num, trainnum)
end

local trainsize = 6327
local testsize = 703
local traindata = torch.Tensor(trainsize, 100, 100)
local testdata = torch.Tensor(testsize, 100, 100)
for i = 1, trainsize do
	traindata[i] = totaldata[i]
end
for i = 1, testsize do
	testdata[i] = totaldata[i+trainsize]
end

torch.save('trainice.t7', traindata)
torch.save('testice.t7', testdata)





