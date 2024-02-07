import json
from dataclasses import dataclass
from crossbar import CrossBarGen

@dataclass
class Resource:
    bram: int
    uram: int
    dsp: int
    luts: int
    reg: int

fixed = {"bram": 0.10, "dsp": 0.0, "ff": 0.05, "lut": 0.15, "uram": 0.0, "hbm": 0.0}
thresh = {"bram": 0.75, "dsp": 0.50, "ff": 0.50, "lut": 0.5, "uram": 0.4, "hbm": 0.9}

def print_resource(config_file, with_ta, with_hb, num_ch, num_c_ch):
    myEst = ResourceEstimate()
    if config_file is not None:
        available = load_config_file(config_file)
        print(f"\nResource Estimation [{num_ch} Channels]")
        utilised = myEst.Total(num_ch, with_ta, with_hb, num_c_ch)
        for key, value in utilised.items():
            value += int(fixed[key] * available[key])
            util = value / available[key]
            print(f"  {key}: {int(value)} [{round(util*100, 2)}%]")
    else:
        print(f"\nResource Estimation [{num_ch} Channels] (only design usage)")
        utilised = myEst.Total(num_ch, with_ta, with_hb, num_c_ch)
        for key, value in utilised.items():
            print(f"  {key}: {int(value)}")


class ResourceEstimate:
    def __init__(self):
        self.num_ch = 0
        self.num_pes = 0
        self.with_ta = 0

    def scale(self, resource, scalar):
        return Resource(
            bram=resource.bram * scalar,
            uram=resource.uram * scalar,
            dsp=resource.dsp * scalar,
            luts=resource.luts * scalar,
            reg=resource.reg * scalar
        )
    
    def add_resrcs(self, resource1, resource2):
        return Resource(
            bram=resource1.bram + resource2.bram,
            uram=resource1.uram + resource2.uram,
            dsp=resource1.dsp + resource2.dsp,
            luts=resource1.luts + resource2.luts,
            reg=resource1.reg + resource2.reg
        )

    def MMAPS(self):
        res = Resource(bram=0, uram=0, dsp=0, luts=87, reg=56)
        res = self.scale(res, self.num_ch + 2 * self.num_c_ch + 1)
        return res
    
    def TreeAdders(self):
        if self.with_ta != 0:
            res = Resource(bram=0, uram=0, dsp=16, luts=2287, reg=2744)
        else:
            res = Resource(bram=0, uram=0, dsp=0, luts=35, reg=125)

        res = self.scale(res, self.num_pes // 2)
        return res
    
    def ResulBuffs(self):
        res = Resource(bram=0, uram=3, dsp=3, luts=838, reg=714)
        res = self.scale(res, self.num_pes)
        return res
    
    def PEGs(self):
        if self.with_hb:
            res = Resource(bram=0, uram=0, dsp=6, luts=2059, reg=1681)
            res = self.add_resrcs(res, Resource(bram=16, uram=0, dsp=0, luts=447, reg=474))
            
        else:
            res = Resource(bram=0, uram=0, dsp=6, luts=800, reg=1092)
            res = self.add_resrcs(res, Resource(bram=16, uram=0, dsp=0, luts=327, reg=606))

        
        res = self.scale(res, self.num_pes/2)
        return res
    
    def DummyRead(self):
        res = Resource(bram=0, uram=0, dsp=5, luts=83, reg=173)
        return res
    
    def Compute_C(self):
        res = Resource(bram=0, uram=0, dsp=130 , luts=6806, reg=9560)
        res = self.scale(res, self.num_c_ch)
        return res
    
    def Arbiter_C(self):
        res = Resource(bram=0, uram=0, dsp=0, luts=99, reg=6)
        res = self.scale(res, self.num_ch*self.num_c_ch)
        res = self.add_resrcs(res, Resource(bram=0, uram=0, dsp=2, luts=229, reg=540))
        return res
    
    def ADD_Blocks(self, num):
        res = Resource(bram=0, uram=0, dsp=2, luts=488, reg=408)
        res = self.scale(res, num)
        return res

    def Switch_Blocks(self, num):
        res = Resource(bram=0, uram=0, dsp=0, luts=129, reg=5)
        res = self.scale(res, num)
        return res
    
    def Crossbar_streams(self, num):
        # if self.with_ta:
        res = Resource(bram=0, uram=0, dsp=0, luts=7, reg=9.5)
        res = self.scale(res, num)
        res = self.add_resrcs(res, Resource(bram=0, uram=0, dsp=0, luts=-75000, reg=39293))
        return res
    
    def CrossBar(self):
        myCB = CrossBarGen(self.num_pes)
        myCB.buildGraph(False)
        res = self.ADD_Blocks(myCB.total_add_blocks)
        res = self.add_resrcs(res, self.Switch_Blocks(myCB.total_sw_blocks))
        res = self.add_resrcs(res, self.Crossbar_streams(myCB.total_stream_depth))
        # print("Total Depth: ", myCB.total_stream_depth)
        return res
        
    
    def Total(self, num_ch, with_ta, with_hb, num_c_ch):
        self.num_ch = num_ch
        self.num_pes = num_ch * 8
        self.with_ta = with_ta
        self.with_hb = with_hb
        self.num_c_ch = num_c_ch
        res = self.MMAPS()
        res = self.add_resrcs(res, self.PEGs())
        res = self.add_resrcs(res, self.DummyRead())
        res = self.add_resrcs(res, self.TreeAdders())
        res = self.add_resrcs(res, self.CrossBar())
        res = self.add_resrcs(res, self.ResulBuffs())
        res = self.add_resrcs(res, self.Arbiter_C())
        res = self.add_resrcs(res, self.Compute_C())
        
        return {"bram": res.bram, "dsp": res.dsp, "uram": res.uram, "lut": res.luts, "ff": res.reg, "hbm": num_ch + 3}
    

def load_config_file(json_file_path):
    try:
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"JSON file '{json_file_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from '{json_file_path}'.")
        return {}


def compute_optimum_num_ch(config_file, with_ta, with_hb, num_c_ch):
    available = load_config_file(config_file)
    max_bram = available["bram"] * thresh["bram"]
    bram_per_ch = 64
    init_guess = max_bram // bram_per_ch
    #num_ch can only be even
    init_guess = int((init_guess // (2*num_c_ch)) * (2*num_c_ch))
    # print("Initial Guess:", init_guess)

    myEst = ResourceEstimate()
    num_ch = init_guess

    while True:
        utilised = myEst.Total(num_ch, with_ta, with_hb, num_c_ch)
        
        satisfied = 0
        for key, value in utilised.items():
            util = value / available[key]
            # print(key, round(util, 2), ((util - thresh[key]) <= 0.0025))
            satisfied += ((util + fixed[key] - thresh[key]) <= 0.01)
        # print()
            # print(satisfied)

        if satisfied >= 5:
            valid = True
            for key, value in utilised.items():
                util = value / available[key]
                # print(key, round(util + fixed[key], 2), (util + fixed[key] - thresh[key]))
                valid &= ((util + fixed[key] - thresh[key]) <= 0.1)
            # print(valid)
            if valid:
                break

            else:
                num_ch -= (2*num_c_ch)


        else:
            num_ch -= (2*num_c_ch)


    print("\nResource Estimation [Optimum]")
    utilised = myEst.Total(num_ch, with_ta, with_hb, num_c_ch)
    for key, value in utilised.items():
        value += int(fixed[key] * available[key])
        util = value / available[key]
        print(f"  {key}: {int(value)} [{round(util*100, 2)}%]")

    return num_ch