import math

class CrossBarGen:
    def __init__(self, size):
        self.graph_dict = {}
        self.b_no = 0 #keep track of numb of blocks
        self.n = size
        self.depth = math.ceil(math.log2(self.n))
        self.idx = [0 for i in range(size)] #keep track of streams
        self.depth_dict = {}
        self.total_stream_depth = 0
        self.total_add_blocks = 0
        self.total_sw_blocks = 0
        
    def updateDepth(self, stream, start, level):
        if start:
            self.depth_dict[stream] = {'start': level, 'end': level, 'depth' : 0}
        else:
            self.depth_dict[stream]['end'] = level 
            
    def computeDepth(self):
        for stream, props in self.depth_dict.items():
            start = props['start']
            end = props['end']
            depth = (end - start) * 2
            for i in range(start, end):
                if (i%2 == 1) and (i < 2*(math.floor(math.log2(self.n)))):
                    depth += 6
            self.depth_dict[stream]['depth'] = depth
            self.total_stream_depth += depth
                
    def updateDict(self, node, incoming_edges, outgoing_edges, color):
        self.graph_dict[node] = {'incoming': incoming_edges, 'outgoing': outgoing_edges, 'color':color}
    
    def addBlock(self, block_name, i, j, color, first, last, level):
        in_list = []
        out_list = []
        
        if block_name.find("ADD") != -1:
            self.total_add_blocks += 1
        
        elif block_name.find("SW") != -1:
            self.total_sw_blocks += 1

        if (first):
            in_list.append(f"FIFO_C_SHF[{i}]")
            in_list.append(f"FIFO_C_SHF[{j}]")
        
        else:
            in_list.append(f"s_{i}_{self.idx[i]}")
            in_list.append(f"s_{j}_{self.idx[j]}")
            self.updateDepth(f"s_{i}_{self.idx[i]}", False, level)
            self.updateDepth(f"s_{j}_{self.idx[j]}", False, level)
            self.idx[i] += 1
            self.idx[j] += 1
            
        
        if (last):
            out_list.append(f"FIFO_C_BUF[{i}]")
            out_list.append(f"FIFO_C_BUF[{j}]")
        
        
        else:
            out_list.append(f"s_{i}_{self.idx[i]}")
            out_list.append(f"s_{j}_{self.idx[j]}")
            self.updateDepth(f"s_{i}_{self.idx[i]}", True, level)
            self.updateDepth(f"s_{j}_{self.idx[j]}", True, level)
            
        block_node_id = f"{self.b_no}.{block_name}.[{level}]"
        self.b_no+=1
        self.updateDict(block_node_id, in_list, out_list, color)
    
    def buildGraph(self, view):
        level = 0
        for d in range(1, self.depth+1):
            div = (1 << d)
            off = div//4 - 1
            first = (d == 1)
            last = (self.n == 2)
            if (off > 0):
                level += 2
            else:
                level += 1
                
            for i in range(div//2, self.n, div):
                if off > 0:
                    self.addBlock("SSW", (i-1)-off, i-1, "green", False, False, level - 2)
                    
                    if (self.n - i) > (div//2 - div//4):
                        self.addBlock("SSW", i, i+off, "green", False, False, level - 2) 
                    
                    else:
                        adjust_off = (1 << (math.ceil(math.log2(self.n - i))-1)) 
                        if adjust_off > 0:
                            self.addBlock("SSW", i, i+adjust_off, "green", False, False, level - 2)
                    
                if d == self.depth:
                    self.addBlock("ADD_X", i-1, i, "red", first, last, level - 1)
                    
                elif ((i >> d) & 1) == 0:
                    self.addBlock("ADD_1", i-1, i, "yellow", first, False, level - 1)
                    
                else:
                    self.addBlock("ADD_0", i-1, i, "yellow", first, False, level - 1)
                    
        for d in range(self.depth, 0, -1):
            div = (1 << d)
            off = div//4 - 1
            first = False
            last = (d == 1)
            
            if (off > 0):
                level += 2
            else:
                level += 1
                
            for i in range(div//2, self.n, div):
                dec = 1
                if off > 0:
                    dec = 2
                if d == self.depth:
                    if(i==div//2):
                        level -= 1
                        
                elif ((i >> d) & 1) == 0:
                    self.addBlock(f"SWB1_{d-1}", i-1, i, "magenta", False, last, level - dec)
                    
                else:
                    self.addBlock(f"SWB0_{d-1}", i-1, i, "magenta", False, last, level - dec)
                    
                
                if off > 0:
                    self.addBlock("SSW", (i-1)-off, i-1, "green", False, last, level - 1)
                    
                    if (self.n - i) > (div//2 - div//4):
                        self.addBlock("SSW", i, i+off, "green", False, last, level - 1) 
                    
                    else:
                        adjust_off = (1 << (math.ceil(math.log2(self.n - i))-1)) 
                        if adjust_off > 0:
                            self.addBlock("SSW", i, i+adjust_off, "green", False, last, level - 1)
            
        self.computeDepth()