class FakeLogger:
  def close(self):
    pass

  def add_scalar(self, tag, value, global_step):
    pass

  def add_histogram(self, tag, values, global_step, bins):
    pass

  def add_image(self, tag, img, global_step):
    pass

  def add_plot(self, tag, figure, global_step):
    pass
