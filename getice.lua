local number = arg[1]

local trueimage = torch.load('all_split_image/trueimage'..number..'.t7')

local iceidx = {}
io.input('ice/ice'..number)
while true do
	local line = io.read("*line")
	if line == nil then break end
	iceidx[#iceidx+1] = tonumber(line)
end
print (iceidx)

local ice = torch.Tensor(#iceidx, 100, 100)

for i=1,#iceidx do
	ice[i] = trueimage[iceidx[i] ]
end

local fn = 'ice/ice'..number..'.t7'
torch.save(fn, ice)

os.execute('scp '..fn..' linzichuan@166.111.131.163:/home/linzichuan/Study/senior_second/particle/show')

