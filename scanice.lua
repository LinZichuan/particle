require 'torch'
require 'nn'
require 'optim'
require 'cutorch'
require 'cunn'

classes = {'1', '2', '3'}
local opt = lapp[[
    -n, --network   (default "network_cuda_ice.t7")    reload pretrained network
    -s, --step      (default '100')                    step of scanning the image
]]
print('<trainer> reloading previously trained network')
model = torch.load(opt.network)
print(model)
criterion = nn.ClassNLLCriterion():cuda()
confusion = optim.ConfusionMatrix(classes)

prob = torch.Tensor(73, 75):fill(-1000)
pos = torch.Tensor(73, 75):fill(0)
maxprob = torch.Tensor(24, 25)
function scan(patchdata, size)
    local res = {}
    local total = 0
    local last = 0
    for i=1, size, 64 do
        xlua.progress(i, size)
        last = i + 63
        if last > size then
            last = size
        end 
        local bs = last - i + 1
        local inputs = torch.CudaTensor(bs, 1, 100, 100)
        for j=1,bs do
            inputs[{j,1}] = patchdata[i+j-1]
        end 
        local outputs = model:forward(inputs)
        for j=1,bs do
            print (outputs[j][1] .. ', ' .. outputs[j][2] .. ', ' .. outputs[j][3])
            if outputs[j][1] > math.max(outputs[j][2], outputs[j][3]) then
                print (i+j-1)
                total = total + 1
                res[#res+1] = i+j-1
				local rr = math.floor((i+j-1-1)/75) + 1
				local cc = (i+j-1-1) % 75 + 1
				prob[rr][cc] = outputs[j][1]
				pos[rr][cc] = i+j-1
            end
        end
    end
    print ('total number is ' .. total)
    return res
end

print  (arg[1])
number = arg[1] --'0003'
local splitimage = torch.load('all_split_image/split_image_stack_'..number..'_cor.mrc.bin.t7') --torch.load('all_split_image/split_image.bin.t7')
local size = splitimage:storage():size()/10000
print (size)
local res = scan(splitimage, size)
--print(res)
print ('res is ' .. #res)
res = {}
for i = 1, 24 do
	for j = 1, 25 do
		local maxp = -1000
		local idx = -1
		local baser = (i-1) * 3
		local basec = (j-1) * 3
		for a = 1, 3 do
			for b = 1, 3 do
				local r = baser + a
				local c = basec + b
				if prob[r][c] > maxp then
					maxp = prob[r][c]
					idx = pos[r][c] --r*75+c
				end
			end
		end
		if idx ~= -1 then res[#res+1] = idx end
	end
end
print ('res is ' .. #res)
trueimage = torch.Tensor(#res, 100, 100)
local inp = assert(io.open('all_split_image/scanres_stack_'..number..'.bin', 'wb'))
local struct = require('struct')

local rr = 3710
local cc = 3838
local side = 100
local step = 50
local row = math.floor((rr-side)/step+1)
local col = math.floor((cc-side)/step+1)
print (row)
print (col)
for i=1,#res do
    trueimage[i] = splitimage[res[i]]:float()
    local indexr = math.floor((res[i]-1) / col) * step
    local indexc = math.floor((res[i]-1) % col) * step
    --print(indexr .. ',' .. indexc)
    inp:write(struct.pack('i4', indexr))
    inp:write(struct.pack('i4', indexc))
end

assert(inp:close())

local fn = 'all_split_image/trueimage'..number..'.t7'
torch.save(fn, trueimage)
os.execute('scp '..fn..' linzichuan@166.111.131.213:~/Study/senior_second/particle/show/trueimage.t7')
os.execute('scp all_split_image/scanres_stack_'..number..'.bin' .. ' linzichuan@166.111.131.213:~/Desktop/qtcreator_5.0.2/build-test1-Desktop_Qt_5_0_2_GCC_64bit-Debug/scanres')
