def generator(data, lookback, min_index, max_index,
              shuffle=False, batch_size=128, step=1):
  import numpy as np 
  
  if max_index is None:
    max_index = len(data) - 1
  i = min_index + lookback
  while 1:
    if shuffle:
      rows = np.random.randint(
              min_index + lookback, max_index, size=batch_size)
    else:
      if i + batch_size >= max_index:
        i = min_index + lookback
      rows = np.arange(i, min(i + batch_size, max_index))
      i += len(rows)
    samples = np.zeros((len(rows),
                        lookback // step,
                        data.shape[-1]))
    targets = np.zeros((len(rows),data.shape[-1]))
    for j, row in enumerate(rows):
      indices = range(rows[j] - lookback, rows[j], step)
      samples[j] = data[indices]
      targets[j] = data[rows[j] ]
    yield samples, targets