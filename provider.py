def sliding_window(input_data, overlap_factor, window_len):
    """ Use sliding window method to segment input data into several bins
        Input:
            input_data: (data_x, data_y)
            overlap_factor:
            window_length: time step length of each sample
        Return:
            output_data: segmented data (data_x, data_y)
    """
    data_x, data_y = input_data
    forward_len = round(window_len*(1 - overlap_factor))
    num_bin = int((data_x.shape[1] - window_len)/forward_len + 1)
    idx = 0
    seg_x = [[[0 for x in range(num_bin*data_x.shape[0])] for y in range(window_len)] for z in range(800)]
    seg_y = [[0 for m in range(num_bin*data_x.shape[0])] for n in range(window_len)]
    for i in range(data_x.shape[0]):
        for j in range(num_bin):
            idx += 1
            seg_x[idx,:,:] = data_x[i,forward_len*j:forward_len*j + window_len,:]
            seg_y[idx,:] = data_y[i,forward_len*j:forward_len*j + window_len]
    return seg_x, seg_y