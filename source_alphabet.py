#!/usr/bin/env python
from gnuradio import gr, blocks
#import mediatools
import numpy as np

class source_alphabet(gr.hier_block2):
    def __init__(self, dtype="discrete", limit=10000, seed=0):
        if(dtype == "discrete"):
            #print(dtype)
            gr.hier_block2.__init__(self, "source_alphabet",
                gr.io_signature(0,0,0),
                gr.io_signature(1,1,gr.sizeof_char))
            np.random.seed(seed)
            rand_bits = np.random.randint(2, size=limit)
            last = blocks.vector_source_b(rand_bits, False, 1, [])
            self.limit = blocks.head(gr.sizeof_char, limit)

        else:   # "type_continuous"
            gr.hier_block2.__init__(self, "source_alphabet",
                gr.io_signature(0,0,0),
                gr.io_signature(1,1,gr.sizeof_float))
            self.src = blocks.wavfile_source('source_material/audio_source.wav', False)
            self.limit = blocks.head(gr.sizeof_float, limit)
            last = self.src
        if dtype=="discrete":   
            self.connect(last, self.limit, self)
        else:
            self.connect(last, self)


if __name__ == "__main__":
    print("QA...")
    tb = gr.top_block()
    src = source_alphabet("continuous", 1000)
    snk = blocks.vector_sink_f()
    tb.connect(src,snk)
    tb.run()
    data = np.array(snk.data(), dtype=np.complex64)
    print(data[0:10])
