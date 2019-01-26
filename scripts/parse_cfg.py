def parsecfg(cfgfile):
	"""
	Takes a configuration file
	Returns a list of blocks. Each blocks describes a block in the neural
	network to be built. Block is represented as a dictionary in the list
	"""
	file = open(cfgfile,'r')
	lines = file.read().split('\n')
	file.close()
	lines = [x for x in lines if len(x)>0 and x[0] != '#']  # get rid of empty line and comments
	lines = [x.strip() for x in lines]  # get rid of white spaces
	
	block = {}
	blocks = []
	
	for line in lines:
		if line[0] == '[':
			if len(block) != 0:
				blocks.append(block)
				block = {}
			block['type'] = line[1:line.find(']')]
		else:
			key,value = line.split('=')
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)  # the last
	
	return  blocks
