from cieran import Cieran
import matplotlib.pyplot as plt
import numpy as np

def draw(self, data,cmap): 
  data2d = np.random.rand(10, 10)
  plt.imshow(data2d, cmap=cmap); plt.show()

class TestSetColor:

  color = '#186E8D'
  cieran = Cieran(draw)

  def test_num_ramps(self):
    self.cieran.set_color(self.color)
    assert len(self.cieran._env.fitted_ramps) >= 80